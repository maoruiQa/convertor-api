package openai

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/songquanpeng/one-api/common/render"

	"github.com/gin-gonic/gin"
	"github.com/songquanpeng/one-api/common"
	"github.com/songquanpeng/one-api/common/conv"
	"github.com/songquanpeng/one-api/common/logger"
	"github.com/songquanpeng/one-api/relay/meta"
	"github.com/songquanpeng/one-api/relay/model"
	"github.com/songquanpeng/one-api/relay/relaymode"
)

const (
	dataPrefix       = "data: "
	done             = "[DONE]"
	dataPrefixLength = len(dataPrefix)
)

func StreamHandler(c *gin.Context, resp *http.Response, relayMode int) (*model.ErrorWithStatusCode, string, *model.Usage) {
	responseText := ""
	scanner := bufio.NewScanner(resp.Body)
	scanner.Split(bufio.ScanLines)
	var usage *model.Usage

	common.SetEventStreamHeaders(c)

	// Chat-only tool-call conversion flag (keep gated in streaming to avoid breaking native streaming semantics)
	m := meta.GetByContext(c)
	convertActive := m != nil && m.PromptToolCallActive && relayMode == relaymode.ChatCompletions

	// Buffers and metadata for conversion streaming
	buffers := make(map[int]string)      // choice index -> aggregated textual content
	sentRole := make(map[int]bool)       // whether role chunk has been sent per index
	var streamId, streamModel string
	var streamCreated int64

	doneRendered := false
	for scanner.Scan() {
		data := scanner.Text()
		if len(data) < dataPrefixLength { // ignore blank line or wrong format
			continue
		}
		if data[:dataPrefixLength] != dataPrefix && data[:dataPrefixLength] != done {
			continue
		}

		// Handle end of stream
		if strings.HasPrefix(data[dataPrefixLength:], done) {
			if convertActive {
				// Flush buffered content as tool_calls chunks
				for idx, buf := range buffers {
					// Try to parse textual tool-call JSON
					tmp := model.Message{Content: buf}
					parsed := convertMessageToolCalls(&tmp)

					// Ensure we have sent the initial role chunk
					if !sentRole[idx] {
						roleChunk := ChatCompletionsStreamResponse{
							Id:      streamId,
							Object:  "chat.completion.chunk",
							Created: streamCreated,
							Model:   streamModel,
							Choices: []ChatCompletionsStreamResponseChoice{{
								Index: idx,
								Delta: model.Message{Role: "assistant"},
							}},
						}
						_ = render.ObjectData(c, roleChunk)
						sentRole[idx] = true
					}

					if parsed && len(tmp.ToolCalls) > 0 {
						// Emit tool_calls chunk with full arguments
						toolChunk := ChatCompletionsStreamResponse{
							Id:      streamId,
							Object:  "chat.completion.chunk",
							Created: streamCreated,
							Model:   streamModel,
							Choices: []ChatCompletionsStreamResponseChoice{{
								Index: idx,
								Delta: model.Message{ToolCalls: tmp.ToolCalls},
							}},
						}
						_ = render.ObjectData(c, toolChunk)

						// Emit finish_reason = tool_calls
						fr := "tool_calls"
						finishChunk := ChatCompletionsStreamResponse{
							Id:      streamId,
							Object:  "chat.completion.chunk",
							Created: streamCreated,
							Model:   streamModel,
							Choices: []ChatCompletionsStreamResponseChoice{{
								Index:        idx,
								Delta:        model.Message{},
								FinishReason: &fr,
							}},
						}
						_ = render.ObjectData(c, finishChunk)
					} else {
						// Fallback: stream the aggregated text content and stop
						if strings.TrimSpace(buf) != "" {
							contentChunk := ChatCompletionsStreamResponse{
								Id:      streamId,
								Object:  "chat.completion.chunk",
								Created: streamCreated,
								Model:   streamModel,
								Choices: []ChatCompletionsStreamResponseChoice{{
									Index: idx,
									Delta: model.Message{Content: buf, Role: "assistant"},
								}},
							}
							_ = render.ObjectData(c, contentChunk)
						}
						fr := "stop"
						finishChunk := ChatCompletionsStreamResponse{
							Id:      streamId,
							Object:  "chat.completion.chunk",
							Created: streamCreated,
							Model:   streamModel,
							Choices: []ChatCompletionsStreamResponseChoice{{
								Index:        idx,
								Delta:        model.Message{},
								FinishReason: &fr,
							}},
						}
						_ = render.ObjectData(c, finishChunk)
					}
				}
				render.Done(c)
				doneRendered = true
				continue
			}
			// default passthrough for non-conversion case
			render.StringData(c, data)
			doneRendered = true
			continue
		}

		switch relayMode {
		case relaymode.ChatCompletions:
			var streamResponse ChatCompletionsStreamResponse
			err := json.Unmarshal([]byte(data[dataPrefixLength:]), &streamResponse)
			if err != nil {
				logger.SysError("error unmarshalling stream response: " + err.Error())
				if !convertActive {
					render.StringData(c, data) // if error happened, pass the data to client
				}
				continue // just ignore the error
			}
			if len(streamResponse.Choices) == 0 && streamResponse.Usage == nil {
				// but for empty choice and no usage, we should not pass it to client, this is for azure
				continue // just ignore empty choice
			}

			// record metadata/usage
			if streamResponse.Id != "" {
				streamId = streamResponse.Id
			}
			if streamResponse.Model != "" {
				streamModel = streamResponse.Model
			}
			if streamResponse.Created != 0 {
				streamCreated = streamResponse.Created
			}
			if streamResponse.Usage != nil {
				usage = streamResponse.Usage
			}

			if convertActive {
				// Intercept: accumulate content and optionally emit role chunk
				for _, choice := range streamResponse.Choices {
					idx := choice.Index
					if choice.Delta.Role != "" && !sentRole[idx] {
						roleChunk := ChatCompletionsStreamResponse{
							Id:      streamId,
							Object:  "chat.completion.chunk",
							Created: streamCreated,
							Model:   streamModel,
							Choices: []ChatCompletionsStreamResponseChoice{{
								Index: idx,
								Delta: model.Message{Role: "assistant"},
							}},
						}
						_ = render.ObjectData(c, roleChunk)
						sentRole[idx] = true
					}
					if choice.Delta.Content != nil {
						buffers[idx] += conv.AsString(choice.Delta.Content)
					}
				}
			} else {
				// default passthrough
				render.StringData(c, data)
				for _, choice := range streamResponse.Choices {
					responseText += conv.AsString(choice.Delta.Content)
				}
			}
		case relaymode.Completions:
			// passthrough for completions
			render.StringData(c, data)
			var streamResponse CompletionsStreamResponse
			err := json.Unmarshal([]byte(data[dataPrefixLength:]), &streamResponse)
			if err != nil {
				logger.SysError("error unmarshalling stream response: " + err.Error())
				continue
			}
			for _, choice := range streamResponse.Choices {
				responseText += choice.Text
			}
		}
	}

	if err := scanner.Err(); err != nil {
		logger.SysError("error reading stream: " + err.Error())
	}

	if !doneRendered {
		render.Done(c)
	}

	err := resp.Body.Close()
	if err != nil {
		return ErrorWrapper(err, "close_response_body_failed", http.StatusInternalServerError), "", nil
	}

return nil, responseText, usage
}

func Handler(c *gin.Context, resp *http.Response, promptTokens int, modelName string) (*model.ErrorWithStatusCode, *model.Usage) {
    var textResponse SlimTextResponse
    responseBody, err := io.ReadAll(resp.Body)
    if err != nil {
        return ErrorWrapper(err, "read_response_body_failed", http.StatusInternalServerError), nil
    }
    err = resp.Body.Close()
    if err != nil {
        return ErrorWrapper(err, "close_response_body_failed", http.StatusInternalServerError), nil
    }
    err = json.Unmarshal(responseBody, &textResponse)
    if err != nil {
        return ErrorWrapper(err, "unmarshal_response_body_failed", http.StatusInternalServerError), nil
    }
    // Opportunistic conversion: try to convert textual JSON tool-calls for ChatCompletions
    converted := false
    if m := meta.GetByContext(c); m != nil && m.Mode == relaymode.ChatCompletions {
        for i := range textResponse.Choices {
            if convertMessageToolCalls(&textResponse.Choices[i].Message) {
                textResponse.Choices[i].FinishReason = "tool_calls"
                converted = true
            }
        }
    }

    // Pass through upstream headers
    for k, v := range resp.Header {
        if len(v) > 0 {
            c.Writer.Header().Set(k, v[0])
        }
    }
    c.Writer.WriteHeader(resp.StatusCode)
    if converted {
        // marshal modified response and send
        newBody, mErr := json.Marshal(textResponse)
        if mErr != nil {
            return ErrorWrapper(mErr, "marshal_modified_response_failed", http.StatusInternalServerError), nil
        }
        // ensure correct content-length
        c.Writer.Header().Set("Content-Length", strconv.Itoa(len(newBody)))
        if _, wErr := c.Writer.Write(newBody); wErr != nil {
            return ErrorWrapper(wErr, "write_modified_response_failed", http.StatusInternalServerError), nil
        }
    } else {
        // Reset response body and pass through
        resp.Body = io.NopCloser(bytes.NewBuffer(responseBody))
        _, err = io.Copy(c.Writer, resp.Body)
        if err != nil {
            return ErrorWrapper(err, "copy_response_body_failed", http.StatusInternalServerError), nil
        }
        err = resp.Body.Close()
        if err != nil {
            return ErrorWrapper(err, "close_response_body_failed", http.StatusInternalServerError), nil
        }
    }

    if textResponse.Usage.TotalTokens == 0 || (textResponse.Usage.PromptTokens == 0 && textResponse.Usage.CompletionTokens == 0) {
        completionTokens := 0
        for _, choice := range textResponse.Choices {
            completionTokens += CountTokenText(choice.Message.StringContent(), modelName)
        }
        textResponse.Usage = model.Usage{
            PromptTokens:     promptTokens,
            CompletionTokens: completionTokens,
            TotalTokens:      promptTokens + completionTokens,
        }
    }
    return nil, &textResponse.Usage
}

// convertMessageToolCalls tries to parse a single JSON object from the assistant message content
// and convert it into structured tool_calls. Returns true if conversion happened.
func convertMessageToolCalls(msg *model.Message) bool {
	content := msg.StringContent()
	if strings.TrimSpace(content) == "" {
		return false
	}
	raw := stripCodeFences(content)
	raw = strings.TrimSpace(raw)
	if !strings.HasPrefix(raw, "{") || !strings.HasSuffix(raw, "}") {
		return false
	}
	// Try parsing {"tool_calls": [...]} shape first
	var wrapper struct {
		ToolCalls []model.Tool `json:"tool_calls"`
	}
	if err := json.Unmarshal([]byte(raw), &wrapper); err == nil && len(wrapper.ToolCalls) > 0 {
		msg.ToolCalls = wrapper.ToolCalls
		// clear content when tool calls are present per OpenAI schema
		msg.Content = ""
		return true
	}
	// Try parsing a single tool-call object: {"type":"function","function":{...}}
	var single struct {
		Type     string         `json:"type"`
		Function model.Function `json:"function"`
	}
	if err := json.Unmarshal([]byte(raw), &single); err == nil && (single.Function.Name != "" || single.Type == "function") {
		if single.Type == "" {
			single.Type = "function"
		}
		msg.ToolCalls = []model.Tool{{
			Type:     single.Type,
			Function: single.Function,
		}}
		msg.Content = ""
		return true
	}
	return false
}

// stripCodeFences removes wrapping markdown code fences like ```json ... ``` if present
func stripCodeFences(s string) string {
	t := strings.TrimSpace(s)
	if !strings.HasPrefix(t, "```") {
		return t
	}
	// drop first line fence
	t = strings.TrimPrefix(t, "```")
	// remove optional language tag at start of line
	if idx := strings.IndexByte(t, '\n'); idx >= 0 {
		t = t[idx+1:]
	}
	// trim trailing fence
	if i := strings.LastIndex(t, "```"); i >= 0 {
		t = t[:i]
	}
	return t
}
