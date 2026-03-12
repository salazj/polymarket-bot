import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { usePortfolio } from "@/hooks/usePortfolio";
import { api } from "@/api/client";
import { formatUSD, formatTimestamp } from "@/lib/utils";
import type { PnLHistoryItem, FillItem } from "@/api/types";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from "recharts";
import { Wallet, ShoppingCart, ArrowRightLeft, TrendingUp, Inbox } from "lucide-react";

type Tab = "positions" | "orders" | "fills" | "pnl";

function EmptyState({ icon: Icon, text }: { icon: React.ElementType; text: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
      <Icon className="h-8 w-8 mb-2 opacity-25" />
      <p className="text-sm">{text}</p>
    </div>
  );
}

export default function Portfolio() {
  const { portfolio, recentOrders } = usePortfolio();
  const [activeTab, setActiveTab] = useState<Tab>("positions");
  const [pnlHistory, setPnlHistory] = useState<PnLHistoryItem[]>([]);
  const [fills, setFills] = useState<FillItem[]>([]);

  useEffect(() => {
    if (activeTab === "pnl") {
      api.getPnLHistory(200).then(setPnlHistory).catch(() => {});
    }
    if (activeTab === "fills") {
      api.getFills(100).then(setFills).catch(() => {});
    }
  }, [activeTab]);

  const tabs: { id: Tab; label: string; icon: React.ElementType }[] = [
    { id: "positions", label: "Positions", icon: Wallet },
    { id: "orders", label: "Orders", icon: ShoppingCart },
    { id: "fills", label: "Fills", icon: ArrowRightLeft },
    { id: "pnl", label: "P&L History", icon: TrendingUp },
  ];

  return (
    <div className="space-y-6">
      {/* Summary strip */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Portfolio</h1>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-1">
          {[
            { label: "Cash", value: formatUSD(portfolio.cash) },
            { label: "Exposure", value: formatUSD(portfolio.total_exposure) },
            { label: "Daily PnL", value: formatUSD(portfolio.daily_pnl), color: portfolio.daily_pnl >= 0 ? "text-emerald-400" : "text-red-400" },
          ].map((s) => (
            <span key={s.label} className="text-sm text-muted-foreground">
              {s.label}: <span className={`font-medium tabular-nums ${s.color || "text-foreground"}`}>{s.value}</span>
            </span>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b overflow-x-auto">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === t.id
                ? "border-primary text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            <t.icon className="h-3.5 w-3.5" />
            {t.label}
          </button>
        ))}
      </div>

      {/* Positions */}
      {activeTab === "positions" && (
        <Card>
          <CardContent className="pt-6">
            {portfolio.positions.length === 0 ? (
              <EmptyState icon={Wallet} text="No open positions" />
            ) : (
              <div className="overflow-x-auto -mx-6">
                <table className="w-full text-sm min-w-[600px]">
                  <thead>
                    <tr className="border-b text-muted-foreground text-xs uppercase tracking-wider">
                      <th className="text-left py-2.5 px-6">Instrument</th>
                      <th className="text-left py-2.5 pr-4">Exchange</th>
                      <th className="text-left py-2.5 pr-4">Side</th>
                      <th className="text-right py-2.5 pr-4">Size</th>
                      <th className="text-right py-2.5 pr-4">Avg Entry</th>
                      <th className="text-right py-2.5 pr-4">Mark</th>
                      <th className="text-right py-2.5 pr-4">Unreal. PnL</th>
                      <th className="text-right py-2.5 px-6">% Return</th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolio.positions.map((p) => {
                      const ret = p.avg_entry_price > 0
                        ? ((p.mark_price - p.avg_entry_price) / p.avg_entry_price) * 100
                        : 0;
                      return (
                        <tr key={p.instrument_id} className="border-b border-border/40 hover:bg-accent/20 transition-colors">
                          <td className="py-2.5 px-6 font-medium">{p.symbol || p.instrument_id}</td>
                          <td className="py-2.5 pr-4 text-muted-foreground text-xs">{p.exchange}</td>
                          <td className="py-2.5 pr-4">
                            <Badge variant={p.side.toLowerCase() === "buy" ? "success" : "destructive"} className="text-[10px]">
                              {p.side}
                            </Badge>
                          </td>
                          <td className="py-2.5 pr-4 text-right tabular-nums">{p.size.toFixed(2)}</td>
                          <td className="py-2.5 pr-4 text-right tabular-nums">{formatUSD(p.avg_entry_price)}</td>
                          <td className="py-2.5 pr-4 text-right tabular-nums">{formatUSD(p.mark_price)}</td>
                          <td className={`py-2.5 pr-4 text-right tabular-nums font-medium ${p.unrealized_pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                            {formatUSD(p.unrealized_pnl)}
                          </td>
                          <td className={`py-2.5 px-6 text-right tabular-nums ${ret >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                            {ret.toFixed(2)}%
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Orders */}
      {activeTab === "orders" && (
        <Card>
          <CardContent className="pt-6">
            {recentOrders.length === 0 ? (
              <EmptyState icon={ShoppingCart} text="No orders" />
            ) : (
              <div className="overflow-x-auto -mx-6">
                <table className="w-full text-sm min-w-[600px]">
                  <thead>
                    <tr className="border-b text-muted-foreground text-xs uppercase tracking-wider">
                      <th className="text-left py-2.5 px-6">Order ID</th>
                      <th className="text-left py-2.5 pr-4">Instrument</th>
                      <th className="text-left py-2.5 pr-4">Side</th>
                      <th className="text-right py-2.5 pr-4">Price</th>
                      <th className="text-right py-2.5 pr-4">Size</th>
                      <th className="text-right py-2.5 pr-4">Filled</th>
                      <th className="text-left py-2.5 pr-4">Status</th>
                      <th className="text-left py-2.5 px-6">Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentOrders.map((o) => (
                      <tr key={o.order_id} className="border-b border-border/40 hover:bg-accent/20 transition-colors">
                        <td className="py-2.5 px-6 font-mono text-xs text-muted-foreground">{o.order_id.slice(0, 8)}</td>
                        <td className="py-2.5 pr-4">{o.instrument_id}</td>
                        <td className="py-2.5 pr-4">{o.side}</td>
                        <td className="py-2.5 pr-4 text-right tabular-nums">{formatUSD(o.price)}</td>
                        <td className="py-2.5 pr-4 text-right tabular-nums">{o.size.toFixed(2)}</td>
                        <td className="py-2.5 pr-4 text-right tabular-nums">{o.filled_size.toFixed(2)}</td>
                        <td className="py-2.5 pr-4">
                          <Badge variant={o.status === "FILLED" ? "success" : o.status === "REJECTED" ? "destructive" : "secondary"} className="text-[10px]">
                            {o.status}
                          </Badge>
                        </td>
                        <td className="py-2.5 px-6 text-muted-foreground text-xs tabular-nums">{formatTimestamp(o.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Fills */}
      {activeTab === "fills" && (
        <Card>
          <CardContent className="pt-6">
            {fills.length === 0 ? (
              <EmptyState icon={ArrowRightLeft} text="No fills yet" />
            ) : (
              <div className="overflow-x-auto -mx-6">
                <table className="w-full text-sm min-w-[440px]">
                  <thead>
                    <tr className="border-b text-muted-foreground text-xs uppercase tracking-wider">
                      <th className="text-left py-2.5 px-6">Order ID</th>
                      <th className="text-right py-2.5 pr-4">Price</th>
                      <th className="text-right py-2.5 pr-4">Size</th>
                      <th className="text-right py-2.5 pr-4">PnL</th>
                      <th className="text-left py-2.5 px-6">Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {fills.map((f, i) => (
                      <tr key={i} className="border-b border-border/40 hover:bg-accent/20 transition-colors">
                        <td className="py-2.5 px-6 font-mono text-xs text-muted-foreground">{f.order_id.slice(0, 8)}</td>
                        <td className="py-2.5 pr-4 text-right tabular-nums">{formatUSD(f.price)}</td>
                        <td className="py-2.5 pr-4 text-right tabular-nums">{f.size.toFixed(2)}</td>
                        <td className={`py-2.5 pr-4 text-right tabular-nums font-medium ${f.pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                          {formatUSD(f.pnl)}
                        </td>
                        <td className="py-2.5 px-6 text-muted-foreground text-xs tabular-nums">{formatTimestamp(f.filled_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* P&L chart */}
      {activeTab === "pnl" && (
        <Card>
          <CardHeader>
            <CardTitle>P&L Over Time</CardTitle>
            <CardDescription>Daily realized and unrealized profit/loss</CardDescription>
          </CardHeader>
          <CardContent>
            {pnlHistory.length === 0 ? (
              <EmptyState icon={TrendingUp} text="No P&L history available" />
            ) : (
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={pnlHistory}>
                  <defs>
                    <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(v) => formatTimestamp(v)}
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={11}
                  />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} />
                  <Tooltip
                    contentStyle={{
                      background: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "0.5rem",
                      fontSize: "12px",
                    }}
                  />
                  <Area type="monotone" dataKey="daily_pnl" stroke="#10b981" fill="url(#pnlGrad)" strokeWidth={2} name="Daily PnL" />
                  <Line type="monotone" dataKey="cash" stroke="#6366f1" strokeWidth={1.5} dot={false} name="Cash" />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
