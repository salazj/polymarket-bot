import { useState, useCallback, useRef } from "react";
import { useWebSocket } from "./useWebSocket";
import type { LogEntry } from "@/api/types";

export function useLogs(level: string = "info") {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const maxLogs = useRef(5000);

  const onMessage = useCallback((data: unknown) => {
    const entry = data as LogEntry;
    if (entry.timestamp && entry.level) {
      setLogs((prev) => {
        const next = [...prev, entry];
        return next.length > maxLogs.current ? next.slice(-maxLogs.current) : next;
      });
    }
  }, []);

  const { connected } = useWebSocket({ url: `/ws/logs?level=${level}`, onMessage });

  const clear = useCallback(() => setLogs([]), []);

  return { logs, connected, clear };
}
