"use client";

import { useState } from "react";
import { marked } from "marked";

const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL;
const API_URL = configuredApiUrl
  ? `${configuredApiUrl.replace(/\/$/, "")}/api/research`.replace(
      "/api/research/api/research",
      "/api/research"
    )
  : "http://127.0.0.1:8000/api/research";
const API_ROOT = API_URL.replace(/\/api\/research\/?$/, "");

const QUICK_PROMPTS = [
  "Impact of Artificial Intelligence on software development",
  "Latest research on Large Language Models",
  "How does Retrieval Augmented Generation work?",
];

export default function Home() {
  const [input, setInput] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [threadId, setThreadId] = useState(null);
  const [requiresApproval, setRequiresApproval] = useState(false);
  const [approvalRequest, setApprovalRequest] = useState("");
  const [feedback, setFeedback] = useState("");

  async function waitForResearch(threadId) {
    while (true) {
      const res = await fetch(
        `${API_ROOT}/api/research/status/${encodeURIComponent(threadId)}`
      );
      const data = await res.json();

      if (!res.ok || data.success === false || data.status === "failed") {
        throw new Error(data.error || "Research failed.");
      }

      if (data.result) {
        const result = data.result;
        setAnswer(result.answer || "");
        setThreadId(result.thread_id || threadId);
        setRequiresApproval(result.requires_approval || false);
        setApprovalRequest(result.approval_request || "");
        return;
      }

      await new Promise((resolve) => setTimeout(resolve, 1500));
    }
  }

  // -----------------------------
  // Start Research
  // -----------------------------

  async function research() {
    if (!input.trim() || loading) return;

    setLoading(true);
    setError("");

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: input,
          thread_id: null,
        }),
      });

      const data = await res.json();

      if (!res.ok || data.success === false) {
        throw new Error(data.error || "Research failed.");
      }

      setThreadId(data.thread_id || null);
      await waitForResearch(data.thread_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  // -----------------------------
  // Approve / Revise
  // -----------------------------

  async function submitReview(approved) {
    if (!threadId || loading) return;

    setLoading(true);
    setError("");

    try {
      const res = await fetch(
        `${API_ROOT}/api/research/approve`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            thread_id: threadId,
            approved,
            feedback: approved ? "" : feedback,
          }),
        }
      );

      const data = await res.json();

      if (!res.ok || data.success === false) {
        throw new Error(data.error || "Review failed.");
      }

      setAnswer(data.answer || "");
      setRequiresApproval(data.requires_approval || false);
      setFeedback("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      research();
    }
  }

  return (
    <main className="min-h-screen bg-[#faf9f6] text-[#1c1b1a] font-sans">
      <header className="border-b border-[#e2ded3] bg-[#faf9f6]">
        <div className="mx-auto max-w-[680px] px-6 pb-10 pt-16 max-[600px]:px-5 max-[600px]:pb-8 max-[600px]:pt-11">
          <span className="font-sans text-[13px] font-semibold tracking-[0.02em] text-[#1f5f5b]">Research desk</span>
          <h1 className="my-2.5 mb-3.5 max-w-[14ch] font-serif text-[40px] font-semibold leading-[1.15] max-[600px]:text-[30px]">Ask a question, get it sourced.</h1>
          <p className="m-0 max-w-[46ch] font-serif text-[18px] leading-[1.55] text-[#5c584f]">
            Draws on web sources, academic papers and news, then hands you
            a report you can review before it's final.
          </p>
        </div>
      </header>

      <div className="mx-auto max-w-[680px] px-6 pb-[100px] pt-10 max-[600px]:px-5">
        {/* Search Box */}
        <section className="border border-[#e2ded3] bg-white p-6">
          <label className="mb-2 block text-[13px] font-medium text-[#5c584f]" htmlFor="query">
            Your question
          </label>
          <textarea
            id="query"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="What do you want to understand?"
          />
          <div className="mt-3.5 flex items-center justify-between max-[600px]:flex-col max-[600px]:items-stretch max-[600px]:gap-3">
            <span className="text-[13px] text-[#5c584f]">⌘/Ctrl + Enter to run</span>
            <button
              className="cursor-pointer border border-[#1c1b1a] bg-[#1c1b1a] px-5 py-[11px] text-[14px] font-medium text-[#faf9f6] hover:border-[#1f5f5b] hover:bg-[#1f5f5b] disabled:cursor-not-allowed disabled:opacity-55 focus-visible:outline-2 focus-visible:outline-[#1f5f5b] focus-visible:outline-offset-2 max-[600px]:w-full"
              onClick={research}
              disabled={loading}
            >
              {loading ? "Researching…" : "Start research"}
            </button>
          </div>
        </section>

        {/* Quick Prompts */}
        <section className="mt-5 flex flex-wrap gap-2">
          {QUICK_PROMPTS.map((prompt) => (
            <button
              key={prompt}
              className="cursor-pointer border border-[#e2ded3] bg-transparent px-3.5 py-2 text-[13px] text-[#5c584f] hover:border-[#1f5f5b] hover:text-[#1f5f5b] disabled:cursor-not-allowed disabled:opacity-55 focus-visible:outline-2 focus-visible:outline-[#1f5f5b] focus-visible:outline-offset-2"
              onClick={() => setInput(prompt)}
            >
              {prompt}
            </button>
          ))}
        </section>

        {/* Error */}
        {error && (
          <div className="mt-6 border-l-[3px] border-[#8c2f1f] bg-[#f7eae7] px-4 py-3.5" role="alert">
            <span className="mb-1 block text-[13px] font-semibold text-[#8c2f1f]">Something went wrong</span>
            <p className="m-0 text-[14px] text-[#1c1b1a]">{error}</p>
          </div>
        )}

        {/* Research Result */}
        {answer && (
          <article className="mt-12 border-t border-[#e2ded3] pt-8">
            <div className="">
              <span className="font-sans text-[13px] font-semibold tracking-[0.02em] text-[#1f5f5b]">Report</span>
              <h2 className="my-1.5 mb-6 font-serif text-[26px] leading-[1.3]">Findings</h2>
            </div>

            <div
              className="font-serif text-[17px] leading-[1.75] text-[#1c1b1a] [&_h1]:font-serif [&_h1]:leading-[1.3] [&_h2]:font-serif [&_h2]:leading-[1.3] [&_h3]:font-serif [&_h3]:leading-[1.3] [&_p]:mb-[1.2em] [&_a]:text-[#1f5f5b] [&_code]:bg-[#e5efed] [&_code]:px-[5px] [&_code]:py-0.5 [&_code]:text-[0.9em]"
              dangerouslySetInnerHTML={{
                __html: marked.parse(answer),
              }}
            />

            {/* HITL */}
            {requiresApproval && (
              <div className="mt-10 border-l-[3px] border-[#1f5f5b] bg-[#e5efed] px-6 py-5">
                <span className="font-sans text-[13px] font-semibold tracking-[0.02em] text-[#1f5f5b]">Needs your review</span>
                <p className="my-2 mb-4 font-serif text-base text-[#1c1b1a]">
                  {approvalRequest ||
                    "Please review the research report before it's finalized."}
                </p>

                <div className="mb-5">
                  <button
                    className="cursor-pointer border border-[#1f5f5b] bg-[#1f5f5b] px-[18px] py-2.5 text-[14px] font-medium text-white hover:bg-[#174a47] disabled:cursor-not-allowed disabled:opacity-55 focus-visible:outline-2 focus-visible:outline-[#1f5f5b] focus-visible:outline-offset-2"
                    onClick={() => submitReview(true)}
                    disabled={loading}
                  >
                    Approve report
                  </button>
                </div>

                <label className="mb-2 block text-[13px] font-medium text-[#5c584f]" htmlFor="feedback">
                  Or request a revision
                </label>
                <textarea
                  className="mb-1 box-border min-h-[110px] w-full resize-y border border-[#e2ded3] bg-white p-3.5 text-[15px] leading-[1.5] text-[#1c1b1a] outline-none focus:border-[#1f5f5b] focus:ring-2 focus:ring-[#1f5f5b]/20"
                  id="feedback"
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                  placeholder="What should change?"
                />
                <button
                  className="mt-3 cursor-pointer border border-[#e2ded3] bg-transparent px-[18px] py-2.5 text-[14px] font-medium text-[#1c1b1a] hover:border-[#1c1b1a] disabled:cursor-not-allowed disabled:opacity-55 focus-visible:outline-2 focus-visible:outline-[#1f5f5b] focus-visible:outline-offset-2"
                  onClick={() => submitReview(false)}
                  disabled={loading || !feedback.trim()}
                >
                  Send revision request
                </button>
              </div>
            )}
          </article>
        )}
      </div>
    </main>
  );

}
