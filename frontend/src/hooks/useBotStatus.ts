import { useState, useCallback } from "react";
import { useWebSocket } from "./useWebSocket";
import type { BotStatus, RiskState } from "@/api/types";

const defaultStatus: BotStatus = {
  running: false, status: "stopped", session_id: "", asset_class: "", exchange: "",
  broker: "", mode: "dry-run", dry_run: true, live_trading: false, uptime_seconds: 0,
  error: null, started_at: null,
};

const defaultRisk: RiskState = {
  halted: false, halt_reason: "", circuit_breaker_tripped: false,
  daily_loss: 0, max_daily_loss: 10, consecutive_losses: 0,
  orders_this_minute: 0, emergency_stop_file_exists: false,
};

export function useBotStatus() {
  const [botStatus, setBotStatus] = useState<BotStatus>(defaultStatus);
  const [riskState, setRiskState] = useState<RiskState>(defaultRisk);

  const onMessage = useCallback((data: unknown) => {
    const msg = data as { type: string; bot?: BotStatus; risk?: RiskState };
    if (msg.type === "status") {
      if (msg.bot) setBotStatus(msg.bot);
      if (msg.risk) setRiskState(msg.risk);
    }
  }, []);

  const { connected } = useWebSocket({ url: "/ws/status", onMessage });

  return { botStatus, riskState, connected };
}
