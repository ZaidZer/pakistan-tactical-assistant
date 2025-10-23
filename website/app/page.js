"use client";
import { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAnalyze = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setError("");
    setAnswer("");

    try {
      const res = await axios.post(`${process.env.NEXT_PUBLIC_API_URL}/analyze`, 
        {question,
      });
      setAnswer(res.data.answer || "No answer returned from backend.");
    } catch (err) {
      console.error(err);
      setError("⚠️ Could not connect to backend. Make sure FastAPI is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-[#0a2e19] via-[#0d3f25] to-[#154d2e] text-white px-4">
      <motion.h1
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
        className="text-4xl md:text-5xl font-extrabold mb-6 text-center text-[#ffcf00] drop-shadow-lg"
      >
        ⚽ Pakistan Tactical Analysis Assistant
      </motion.h1>

      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="max-w-2xl w-full bg-white/10 backdrop-blur-lg p-6 rounded-2xl shadow-lg border border-white/20"
      >
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask your tactical question... e.g., How should Pakistan defend Myanmar’s wide play?"
          className="w-full p-4 rounded-xl bg-[#0f2617] text-white border border-white/20 focus:outline-none focus:ring-2 focus:ring-[#ffcf00]"
          rows={4}
        />
        <button
          onClick={handleAnalyze}
          disabled={loading}
          className="mt-4 w-full py-3 rounded-xl font-semibold text-black bg-[#ffcf00] hover:bg-[#ffd633] active:bg-[#e6ba00] transition"
        >
          {loading ? "Analyzing..." : "Generate Tactical Report"}
        </button>
      </motion.div>

      {error && (
        <div className="mt-4 text-red-400 font-medium">{error}</div>
      )}

      {answer && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="max-w-3xl mt-8 bg-white/10 backdrop-blur-lg p-6 rounded-2xl shadow-lg border border-white/20 text-left"
        >
          <h2 className="text-xl font-semibold mb-3 text-[#ffcf00]">
            🧠 Tactical Report
          </h2>
          <p className="whitespace-pre-wrap leading-relaxed text-gray-100">
            {answer}
          </p>
        </motion.div>
      )}

      <footer className="mt-10 text-sm text-gray-400">
        © 2025 Pakistan Tactical AI – Powered by GPT-4o
      </footer>
    </main>
  );
}
