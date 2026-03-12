import { useState, useRef, useEffect, useCallback } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ConnectionDot } from "@/components/ConnectionDot";
import { useLogs } from "@/hooks/useLogs";
import { formatTimestamp } from "@/lib/utils";
import { Download, Trash2, ArrowDown, Pause, Play } from "lucide-react";

const LEVEL_COLORS: Record<string, string> = {
  debug: "text-zinc-500",
  info: "text-foreground",
  warning: "text-amber-400",
  error: "text-red-400",
  critical: "text-red-500 font-bold",
};

const LEVEL_BADGE: Record<string, "secondary" | "default" | "warning" | "destructive"> = {
  debug: "secondary",
  info: "default",
  warning: "warning",
  error: "destructive",
  critical: "destructive",
};

export default function Logs() {
  const [level, setLevel] = useState("info");
  const [search, setSearch] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const { logs, connected, clear } = useLogs(level);
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoScroll) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, autoScroll]);

  const handleScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    if (!nearBottom && autoScroll) setAutoScroll(false);
    if (nearBottom && !autoScroll) setAutoScroll(true);
  }, [autoScroll]);

  const filtered = search
    ? logs.filter(
        (l) =>
          l.event.toLowerCase().includes(search.toLowerCase()) ||
          l.logger.toLowerCase().includes(search.toLowerCase()) ||
          JSON.stringify(l.data).toLowerCase().includes(search.toLowerCase())
      )
    : logs;

  const exportLogs = () => {
    const blob = new Blob([JSON.stringify(filtered, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `salazar-trader-logs-${new Date().toISOString().slice(0, 19)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Live Logs</h1>
          <div className="flex items-center gap-2 text-sm text-muted-foreground mt-0.5">
            <ConnectionDot connected={connected} />
            <span>{connected ? "Streaming" : "Disconnected"}</span>
            <span className="text-muted-foreground/40">·</span>
            <span>{filtered.length} entries</span>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <select
            value={level}
            onChange={(e) => setLevel(e.target.value)}
            className="bg-input border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
          >
            <option value="debug">DEBUG+</option>
            <option value="info">INFO+</option>
            <option value="warning">WARNING+</option>
            <option value="error">ERROR+</option>
          </select>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search logs..."
            className="bg-input border rounded-md px-3 py-2 text-sm w-40 focus:outline-none focus:ring-2 focus:ring-ring"
          />
          <div className="flex items-center gap-1 border rounded-md bg-input/50 p-0.5">
            <Button
              variant={autoScroll ? "default" : "ghost"}
              size="icon"
              className="h-8 w-8"
              onClick={() => { setAutoScroll(true); bottomRef.current?.scrollIntoView(); }}
              title="Auto-scroll"
            >
              <ArrowDown className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant={!autoScroll ? "default" : "ghost"}
              size="icon"
              className="h-8 w-8"
              onClick={() => setAutoScroll(false)}
              title="Pause scroll"
            >
              <Pause className="h-3.5 w-3.5" />
            </Button>
          </div>
          <Button variant="ghost" size="icon" className="h-9 w-9" onClick={exportLogs} title="Export">
            <Download className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" className="h-9 w-9" onClick={clear} title="Clear">
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Log stream */}
      <Card className="overflow-hidden">
        <CardContent className="p-0">
          <div
            ref={containerRef}
            onScroll={handleScroll}
            className="h-[calc(100vh-220px)] sm:h-[calc(100vh-200px)] overflow-y-auto font-mono text-xs leading-relaxed"
          >
            {filtered.length === 0 ? (
              <div className="p-12 text-center text-muted-foreground">
                <Play className="h-8 w-8 mx-auto mb-3 opacity-30" />
                <p>Waiting for log events...</p>
                <p className="text-xs mt-1">Logs will appear here when the bot is running</p>
              </div>
            ) : (
              <div className="divide-y divide-border/20">
                {filtered.map((entry, i) => (
                  <div
                    key={i}
                    className={`flex gap-2 px-3 py-1.5 hover:bg-accent/20 transition-colors ${LEVEL_COLORS[entry.level] || ""}`}
                  >
                    <span className="text-muted-foreground/60 shrink-0 w-[72px] tabular-nums">
                      {formatTimestamp(entry.timestamp)}
                    </span>
                    <Badge
                      variant={LEVEL_BADGE[entry.level] || "secondary"}
                      className="shrink-0 h-[18px] text-[9px] px-1.5 min-w-[44px] justify-center"
                    >
                      {entry.level.toUpperCase()}
                    </Badge>
                    <span className="flex-1 break-all">
                      <span className="font-semibold">{entry.event}</span>
                      {Object.keys(entry.data).length > 0 && (
                        <span className="text-muted-foreground ml-2">
                          {Object.entries(entry.data)
                            .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`)
                            .join(" ")}
                        </span>
                      )}
                    </span>
                  </div>
                ))}
                <div ref={bottomRef} />
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
