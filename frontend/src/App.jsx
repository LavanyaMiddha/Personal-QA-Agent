import { useState, useRef, useEffect } from "react";

const API_URL = "http://localhost:8000";

function generateThreadId() {
  return `session-${Date.now()}`;
}

function SourceBadge({ source }) {
  const filename = source.split(/[\\/]/).pop();
  return (
    <span style={{
      display: "inline-block",
      background: "rgba(99,211,168,0.12)",
      border: "1px solid rgba(99,211,168,0.3)",
      color: "#63d3a8",
      borderRadius: "4px",
      padding: "2px 8px",
      fontSize: "11px",
      fontFamily: "monospace",
      marginRight: "6px",
      marginTop: "6px",
    }}>
      📄 {filename}
    </span>
  );
}

function Message({ msg }) {
  const isUser = msg.role === "user";

  // Extract sources from answer (lines starting with "Source:")
  const sources = [];
  if (!isUser && msg.content) {
    const lines = msg.content.split("\n");
    lines.forEach(line => {
      const match = line.match(/Source:\s*(.+)/i);
      if (match) sources.push(match[1].trim());
    });
  }

  return (
    <div style={{
      display: "flex",
      justifyContent: isUser ? "flex-end" : "flex-start",
      marginBottom: "20px",
      gap: "12px",
      alignItems: "flex-start",
    }}>
      {!isUser && (
        <div style={{
          width: "32px", height: "32px", borderRadius: "50%",
          background: "linear-gradient(135deg, #63d3a8, #3b82f6)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "14px", flexShrink: 0, marginTop: "2px",
        }}>🤖</div>
      )}
      <div style={{
        maxWidth: "72%",
        background: isUser
          ? "linear-gradient(135deg, #3b82f6, #6366f1)"
          : "rgba(255,255,255,0.05)",
        border: isUser ? "none" : "1px solid rgba(255,255,255,0.08)",
        borderRadius: isUser ? "18px 18px 4px 18px" : "4px 18px 18px 18px",
        padding: "14px 18px",
        color: "#f1f5f9",
        fontSize: "14px",
        lineHeight: "1.7",
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
      }}>
        {msg.content}
        {sources.length > 0 && (
          <div style={{ marginTop: "10px", borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: "10px" }}>
            {sources.map((s, i) => <SourceBadge key={i} source={s} />)}
          </div>
        )}
      </div>
      {isUser && (
        <div style={{
          width: "32px", height: "32px", borderRadius: "50%",
          background: "rgba(255,255,255,0.1)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "14px", flexShrink: 0, marginTop: "2px",
        }}>👤</div>
      )}
    </div>
  );
}

function TypingIndicator() {
  return (
    <div style={{ display: "flex", alignItems: "flex-start", gap: "12px", marginBottom: "20px" }}>
      <div style={{
        width: "32px", height: "32px", borderRadius: "50%",
        background: "linear-gradient(135deg, #63d3a8, #3b82f6)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: "14px", flexShrink: 0,
      }}>🤖</div>
      <div style={{
        background: "rgba(255,255,255,0.05)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: "4px 18px 18px 18px",
        padding: "16px 20px",
        display: "flex", gap: "6px", alignItems: "center",
      }}>
        {[0, 1, 2].map(i => (
          <div key={i} style={{
            width: "7px", height: "7px", borderRadius: "50%",
            background: "#63d3a8",
            animation: "bounce 1.2s infinite",
            animationDelay: `${i * 0.2}s`,
          }} />
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hello! I'm your personal document assistant. Ask me anything about your uploaded PDFs and I'll retrieve relevant context to answer your questions.",
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [threadId] = useState(generateThreadId);
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function sendMessage() {
    const query = input.trim();
    if (!query || loading) return;

    setInput("");
    setError(null);
    setMessages(prev => [...prev, { role: "user", content: query }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, thread_id: threadId }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Request failed");
      }

      const data = await res.json();
      setMessages(prev => [...prev, { role: "assistant", content: data.answer }]);
    } catch (e) {
      setError(e.message);
      setMessages(prev => [...prev, {
        role: "assistant",
        content: `⚠️ Error: ${e.message}. Please check that the FastAPI server is running.`,
      }]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <>
      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0d1117; font-family: 'Georgia', serif; }
        @keyframes bounce {
          0%, 60%, 100% { transform: translateY(0); }
          30% { transform: translateY(-6px); }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        textarea:focus { outline: none; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
      `}</style>

      <div style={{
        minHeight: "100vh",
        background: "#0d1117",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}>

        {/* Header */}
        <div style={{
          width: "100%",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
          padding: "18px 32px",
          display: "flex",
          alignItems: "center",
          gap: "12px",
          background: "rgba(255,255,255,0.02)",
          backdropFilter: "blur(10px)",
          position: "sticky",
          top: 0,
          zIndex: 10,
        }}>
          <div style={{
            width: "36px", height: "36px", borderRadius: "10px",
            background: "linear-gradient(135deg, #63d3a8, #3b82f6)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "18px",
          }}>📚</div>
          <div>
            <div style={{ color: "#f1f5f9", fontWeight: "600", fontSize: "15px", letterSpacing: "0.02em" }}>
              Personal QA Agent
            </div>
            <div style={{ color: "#64748b", fontSize: "12px" }}>
              RAG-powered document assistant
            </div>
          </div>
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "6px" }}>
            <div style={{ width: "7px", height: "7px", borderRadius: "50%", background: "#63d3a8" }} />
            <span style={{ color: "#63d3a8", fontSize: "12px" }}>Connected</span>
          </div>
        </div>

        {/* Chat area */}
        <div style={{
          flex: 1,
          width: "100%",
          maxWidth: "780px",
          padding: "32px 24px",
          overflowY: "auto",
          animation: "fadeIn 0.4s ease",
        }}>
          {messages.map((msg, i) => (
            <Message key={i} msg={msg} />
          ))}
          {loading && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div style={{
          width: "100%",
          maxWidth: "780px",
          padding: "16px 24px 28px",
        }}>
          <div style={{
            display: "flex",
            gap: "12px",
            alignItems: "flex-end",
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: "16px",
            padding: "12px 16px",
            transition: "border-color 0.2s",
          }}>
            <textarea
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents..."
              rows={1}
              style={{
                flex: 1,
                background: "transparent",
                border: "none",
                color: "#f1f5f9",
                fontSize: "14px",
                lineHeight: "1.6",
                resize: "none",
                fontFamily: "inherit",
                maxHeight: "140px",
                overflowY: "auto",
              }}
              onInput={e => {
                e.target.style.height = "auto";
                e.target.style.height = e.target.scrollHeight + "px";
              }}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || loading}
              style={{
                width: "36px", height: "36px", borderRadius: "10px",
                background: input.trim() && !loading
                  ? "linear-gradient(135deg, #63d3a8, #3b82f6)"
                  : "rgba(255,255,255,0.06)",
                border: "none",
                cursor: input.trim() && !loading ? "pointer" : "not-allowed",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: "16px",
                transition: "all 0.2s",
                flexShrink: 0,
              }}
            >
              ➤
            </button>
          </div>
          <div style={{ textAlign: "center", color: "#334155", fontSize: "11px", marginTop: "10px" }}>
            Enter to send · Shift+Enter for new line · Thread: {threadId.slice(0, 20)}...
          </div>
        </div>
      </div>
    </>
  );
}