// Where the FastAPI backend lives. Set NEXT_PUBLIC_API_URL at build time (see .env.example).
// Default is 127.0.0.1, not "localhost": uvicorn binds IPv4 only, and some browsers resolve
// localhost to IPv6 (::1) first without falling back, which looks like "connection refused".
export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

// http -> ws, https -> wss
export const WS_URL = API_URL.replace(/^http/, "ws");

export const ACTIVITY_COLORS = {
  series: "#2a78d6", // the signal line
  flagged: "#d03b3b", // detector flagged this reading (status: critical)
  injected: "rgba(250, 178, 25, 0.28)", // a fault was really injected here (status: warning)
};
