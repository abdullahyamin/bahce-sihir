import { useRef, useState } from "react";
import { sendQuery } from "../api";
import "./ChatPanel.css";

const SOURCES_LINE_RE = /\n+SOURCES:\s*([^\n]+)\s*$/i;

function splitCitation(content) {
  const match = content.match(SOURCES_LINE_RE);
  if (!match) return { answer: content, citation: null };
  return { answer: content.slice(0, match.index).trim(), citation: match[1].trim() };
}

const WELCOME = {
  role: "assistant",
  content:
    "Hello! I'm your Bahçeşehir University assistant. I can answer your questions about regulations, directives, and procedures.",
  sources: [],
};

export default function ChatPanel() {
  const [messages, setMessages] = useState([WELCOME]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bodyRef = useRef(null);

  async function handleSend() {
    const query = input.trim();
    if (!query || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setInput("");
    setLoading(true);

    try {
      const result = await sendQuery(query);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: result.answer, sources: result.sources },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Sorry, something went wrong. Please make sure the server is running and try again.",
          sources: [],
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
      requestAnimationFrame(() => {
        bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight, behavior: "smooth" });
      });
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <div className="seal">BAU</div>
        <div className="name">
          University Assistant
          <span>Bahçeşehir University</span>
        </div>
      </div>

      <div className="chat-body" ref={bodyRef}>
        {messages.map((m, i) => {
          const { answer, citation } = m.role === "assistant" ? splitCitation(m.content) : { answer: m.content, citation: null };
          return (
          <div key={i} className="message-group">
            <div className={`bubble ${m.role}`}>{answer}</div>
            {citation && <div className="citation-footer">{citation}</div>}
            {m.sources && m.sources.length > 0 && (
              <div className="sources">
                {m.sources.map((s, j) => (
                  <div key={j} className="source-chip">
                    {s.source_file.replace(/\.pdf$|\.docx$/i, "")}
                    {s.article_no ? ` — Article ${s.article_no}` : ""}
                  </div>
                ))}
              </div>
            )}
          </div>
          );
        })}
        {loading && (
          <div className="bubble assistant loading">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </div>
        )}
      </div>

      <div className="chat-input">
        <textarea
          className="ph"
          placeholder="Ask a question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
        />
        <button className="send" onClick={handleSend} disabled={loading || !input.trim()}>
          &#8594;
        </button>
      </div>
    </div>
  );
}
