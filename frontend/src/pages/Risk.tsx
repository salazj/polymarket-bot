import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/components/ui/toast";
import { useBotStatus } from "@/hooks/useBotStatus";
import { api } from "@/api/client";
import { formatUSD } from "@/lib/utils";
import { ShieldAlert, ShieldCheck, OctagonX, RotateCcw, AlertTriangle, Gauge } from "lucide-react";

export default function Risk() {
  const { riskState, botStatus } = useBotStatus();
  const [confirmEmergency, setConfirmEmergency] = useState(false);
  const [loading, setLoading] = useState(false);
  const { addToast } = useToast();

  const handleResetBreaker = async () => {
    setLoading(true);
    try {
      await api.resetBreaker();
      addToast({ title: "Circuit breaker reset", variant: "success" });
    } catch (e: unknown) {
      addToast({ title: "Failed to reset", description: (e as Error).message, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  const handleEmergencyStop = async () => {
    if (!confirmEmergency) {
      setConfirmEmergency(true);
      return;
    }
    setLoading(true);
    try {
      await api.emergencyStop();
      setConfirmEmergency(false);
      addToast({ title: "Emergency stop activated", description: "All trading halted", variant: "destructive" });
    } catch (e: unknown) {
      addToast({ title: "Emergency stop failed", description: (e as Error).message, variant: "destructive" });
    } finally {
      setLoading(false);
    }
  };

  const lossRatio = riskState.max_daily_loss > 0
    ? Math.abs(riskState.daily_loss) / riskState.max_daily_loss
    : 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Risk Controls</h1>
        <p className="text-muted-foreground text-sm">Monitor and manage trading risk in real time</p>
      </div>

      {/* Live trading warning */}
      {botStatus.live_trading && (
        <div className="rounded-lg border border-red-700/50 bg-red-950/30 p-4 flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-red-400 shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold text-red-400 text-sm">Live Trading Active</div>
            <p className="text-xs text-red-300/70 mt-1">
              All three safety gates are open. Real orders are being submitted.
            </p>
          </div>
        </div>
      )}

      {/* Risk overview */}
      <Card className={riskState.halted ? "border-red-700/50" : ""}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {riskState.halted ? (
              <ShieldAlert className="h-5 w-5 text-red-400" />
            ) : (
              <ShieldCheck className="h-5 w-5 text-emerald-400" />
            )}
            System Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Status */}
            <div className="p-4 rounded-lg border bg-card">
              <div className="text-xs text-muted-foreground uppercase tracking-wider mb-2">Status</div>
              {riskState.halted ? (
                <Badge variant="destructive" className="text-sm px-3 py-1">HALTED</Badge>
              ) : (
                <Badge variant="success" className="text-sm px-3 py-1">Normal</Badge>
              )}
              {riskState.halt_reason && (
                <p className="text-xs text-red-400 mt-2">{riskState.halt_reason}</p>
              )}
            </div>

            {/* Circuit Breaker */}
            <div className="p-4 rounded-lg border bg-card">
              <div className="text-xs text-muted-foreground uppercase tracking-wider mb-2">Circuit Breaker</div>
              <div className="text-lg font-semibold">
                {riskState.circuit_breaker_tripped ? (
                  <span className="text-red-400">TRIPPED</span>
                ) : (
                  <span className="text-emerald-400">OK</span>
                )}
              </div>
            </div>

            {/* Daily Loss */}
            <div className="p-4 rounded-lg border bg-card">
              <div className="text-xs text-muted-foreground uppercase tracking-wider mb-2">Daily Loss</div>
              <div className={`text-lg font-semibold tabular-nums ${lossRatio > 0.5 ? "text-red-400" : ""}`}>
                {formatUSD(Math.abs(riskState.daily_loss))}
                <span className="text-xs text-muted-foreground font-normal ml-1">/ {formatUSD(riskState.max_daily_loss)}</span>
              </div>
              <div className="mt-2 h-2 rounded-full bg-secondary overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    lossRatio > 0.8 ? "bg-red-500" : lossRatio > 0.5 ? "bg-amber-500" : "bg-emerald-500"
                  }`}
                  style={{ width: `${Math.min(lossRatio * 100, 100)}%` }}
                />
              </div>
            </div>

            {/* Consecutive Losses */}
            <div className="p-4 rounded-lg border bg-card">
              <div className="text-xs text-muted-foreground uppercase tracking-wider mb-2">Consecutive Losses</div>
              <div className={`text-lg font-semibold tabular-nums ${riskState.consecutive_losses > 3 ? "text-amber-400" : ""}`}>
                {riskState.consecutive_losses}
              </div>
            </div>

            {/* Orders / Min */}
            <div className="p-4 rounded-lg border bg-card">
              <div className="text-xs text-muted-foreground uppercase tracking-wider mb-2">Orders / Minute</div>
              <div className="text-lg font-semibold tabular-nums">{riskState.orders_this_minute}</div>
            </div>

            {/* Emergency File */}
            <div className="p-4 rounded-lg border bg-card">
              <div className="text-xs text-muted-foreground uppercase tracking-wider mb-2">Emergency Stop File</div>
              <div className="text-lg font-semibold">
                {riskState.emergency_stop_file_exists ? (
                  <span className="text-red-400">EXISTS</span>
                ) : (
                  <span className="text-muted-foreground/60">Not present</span>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Actions */}
      <div className="grid sm:grid-cols-2 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <RotateCcw className="h-4 w-4" />
              Reset Circuit Breaker
            </CardTitle>
            <CardDescription>Re-enable trading after a circuit breaker trip</CardDescription>
          </CardHeader>
          <CardContent>
            <Button
              variant="outline"
              onClick={handleResetBreaker}
              disabled={loading || !botStatus.running}
            >
              Reset Breaker
            </Button>
          </CardContent>
        </Card>

        <Card className="border-red-800/40">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base text-red-400">
              <OctagonX className="h-4 w-4" />
              Emergency Stop
            </CardTitle>
            <CardDescription>Immediately halt all trading and cancel open orders</CardDescription>
          </CardHeader>
          <CardContent>
            {confirmEmergency ? (
              <div className="space-y-3">
                <p className="text-sm text-red-400 font-semibold">
                  Are you sure? This will cancel all orders and halt the bot.
                </p>
                <div className="flex gap-2">
                  <Button variant="destructive" onClick={handleEmergencyStop} disabled={loading}>
                    Confirm Emergency Stop
                  </Button>
                  <Button variant="ghost" onClick={() => setConfirmEmergency(false)}>
                    Cancel
                  </Button>
                </div>
              </div>
            ) : (
              <Button
                variant="destructive"
                onClick={handleEmergencyStop}
                disabled={loading || !botStatus.running}
              >
                Emergency Stop
              </Button>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
