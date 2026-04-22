import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
});

export async function queryRAG(question, dataset = "both", top_k = 5) {
  const response = await api.post("/query", { question, dataset, top_k });
  return response.data;
}

export async function getCollections() {
  const response = await api.get("/collections");
  return response.data;
}

export async function getHealth() {
  const response = await api.get("/health");
  return response.data;
}
// Wed Apr 15 19:36:39 EDT 2026

export async function getSuggestions(question, dataset = "both") {
  const response = await api.post("/suggestions", { question, dataset });
  return response.data;
}
