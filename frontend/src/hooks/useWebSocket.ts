import { useEffect, useRef, useState, useCallback } from "react";

interface UseWebSocketOptions {
  url: string;
  onMessage?: (data: unknown) => void;
  reconnectInterval?: number;
  enabled?: boolean;
}

export function useWebSocket({ url, onMessage, reconnectInterval = 3000, enabled = true }: UseWebSocketOptions) {
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();

  const connect = useCallback(() => {
    if (!enabled) return;
    try {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const host = window.location.host;
      const fullUrl = url.startsWith("ws") ? url : `${protocol}//${host}${url}`;
      const ws = new WebSocket(fullUrl);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        reconnectTimer.current = setTimeout(connect, reconnectInterval);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          onMessage?.(data);
        } catch {
          // ignore non-JSON messages
        }
      };
    } catch {
      reconnectTimer.current = setTimeout(connect, reconnectInterval);
    }
  }, [url, onMessage, reconnectInterval, enabled]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { connected };
}
