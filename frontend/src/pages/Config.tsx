import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/components/ui/toast";
import { api } from "@/api/client";
import { DEFAULT_RUN_CONFIG, type RunConfig, type StrategyInfo } from "@/api/types";
import {
  BarChart3,
  TrendingUp,
  Settings,
  AlertTriangle,
  Brain,
  Save,
  FolderOpen,
  Trash2,
  Layers,
  Shield,
  Target,
  Globe,
  Cpu,
  Check,
} from "lucide-react";

interface Preset {
  name: string;
  config: RunConfig;
  created_at?: string;
}

export default function Config() {
  const [config, setConfig] = useState<RunConfig>({ ...DEFAULT_RUN_CONFIG });
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [presetName, setPresetName] = useState("");
  const [showPresets, setShowPresets] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [validated, setValidated] = useState(false);
  const { addToast } = useToast();

  useEffect(() => {
    api.getConfig().then((c) => setConfig({ ...DEFAULT_RUN_CONFIG, ...c })).catch(() => {});
    api.getStrategies().then(setStrategies).catch(() => {});
    api.getPresets().then(setPresets).catch(() => {});
  }, []);

  const update = (partial: Partial<RunConfig>) => {
    setConfig((prev) => ({ ...prev, ...partial }));
    setValidated(false);
    setErrors([]);
    setWarnings([]);
  };

  const handleValidate = async () => {
    try {
      const result = await api.validateConfig(config);
      setErrors(result.errors);
      setWarnings(result.warnings);
      setValidated(result.valid);
      if (result.valid) {
        addToast({ title: "Configuration valid", variant: "success" });
      }
      return result.valid;
    } catch (e: unknown) {
      setErrors([(e as Error).message]);
      return false;
    }
  };

  const handleStart = async () => {
    setLoading(true);
    const valid = await handleValidate();
    if (!valid) {
      setLoading(false);
      return;
    }
    try {
      await api.startBot(config);
      addToast({ title: "Bot started", description: `${config.asset_class} · ${config.dry_run ? "dry-run" : "LIVE"}`, variant: "success" });
    } catch (e: unknown) {
      addToast({ title: "Failed to start", description: (e as Error).message, variant: "destructive" });
      setErrors([(e as Error).message]);
    } finally {
      setLoading(false);
    }
  };

  const handleSavePreset = async () => {
    if (!presetName.trim()) return;
    try {
      await api.savePreset(presetName.trim(), config);
      addToast({ title: "Preset saved", description: presetName, variant: "success" });
      setPresetName("");
      api.getPresets().then(setPresets).catch(() => {});
    } catch (e: unknown) {
      addToast({ title: "Failed to save preset", description: (e as Error).message, variant: "destructive" });
    }
  };

  const handleLoadPreset = (preset: Preset) => {
    setConfig({ ...DEFAULT_RUN_CONFIG, ...preset.config });
    setShowPresets(false);
    setValidated(false);
    setErrors([]);
    setWarnings([]);
    addToast({ title: "Preset loaded", description: preset.name, variant: "default" });
  };

  const filteredStrategies = strategies.filter((s) =>
    config.asset_class === "equities" ? s.asset_class === "equities" : s.asset_class === "prediction_markets"
  );

  const isLive = !config.dry_run && config.enable_live_trading && config.live_trading_acknowledged;

  type ModeKey = "polymarket" | "kalshi" | "stocks";
  const selectedMode: ModeKey =
    config.asset_class === "equities" ? "stocks" : (config.exchange as ModeKey);

  const setMode = (mode: ModeKey) => {
    if (mode === "stocks") {
      update({ asset_class: "equities", broker: "alpaca" });
    } else {
      update({ asset_class: "prediction_markets", exchange: mode });
    }
  };

  const modeCards: { key: ModeKey; icon: React.ReactNode; label: string; desc: string; color: string }[] = [
    {
      key: "polymarket",
      icon: <Globe className="h-7 w-7" />,
      label: "Polymarket",
      desc: "Decentralized prediction market on Polygon",
      color: "border-indigo-500 bg-indigo-950/20",
    },
    {
      key: "kalshi",
      icon: <Target className="h-7 w-7" />,
      label: "Kalshi",
      desc: "Regulated event contracts exchange",
      color: "border-violet-500 bg-violet-950/20",
    },
    {
      key: "stocks",
      icon: <TrendingUp className="h-7 w-7" />,
      label: "Stocks",
      desc: "Trade equities via Alpaca",
      color: "border-emerald-500 bg-emerald-950/20",
    },
  ];

  return (
    <div className="space-y-8">
      {/* Header + presets */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Run Configuration</h1>
          <p className="text-muted-foreground text-sm">Configure and start a trading session</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => setShowPresets(!showPresets)}>
            <FolderOpen className="h-4 w-4 mr-1.5" /> Presets
          </Button>
        </div>
      </div>

      {/* Preset drawer */}
      {showPresets && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Saved Presets</CardTitle>
            <CardDescription>Load or save configuration presets</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {presets.length === 0 ? (
              <p className="text-sm text-muted-foreground">No saved presets yet</p>
            ) : (
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {presets.map((p) => (
                  <button
                    key={p.name}
                    onClick={() => handleLoadPreset(p)}
                    className="flex items-center gap-3 rounded-lg border p-3 text-left hover:bg-accent/30 transition-colors"
                  >
                    <Layers className="h-4 w-4 text-muted-foreground shrink-0" />
                    <div className="min-w-0">
                      <div className="text-sm font-medium truncate">{p.name}</div>
                      <div className="text-[11px] text-muted-foreground">
                        {p.config.asset_class === "equities" ? "Stocks" : p.config.exchange} · {p.config.dry_run ? "Dry Run" : "Live"}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
            <div className="flex gap-2 pt-2 border-t">
              <input
                type="text"
                value={presetName}
                onChange={(e) => setPresetName(e.target.value)}
                placeholder="Preset name..."
                className="flex-1 bg-input border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              />
              <Button variant="outline" size="sm" onClick={handleSavePreset} disabled={!presetName.trim()}>
                <Save className="h-4 w-4 mr-1" /> Save Current
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ─── Mode selection ─── */}
      <div>
        <h2 className="text-xs font-semibold text-muted-foreground mb-3 uppercase tracking-widest">
          Select Mode
        </h2>
        <div className="grid sm:grid-cols-3 gap-4">
          {modeCards.map((m) => (
            <button
              key={m.key}
              onClick={() => setMode(m.key)}
              className={`group relative p-5 rounded-xl border-2 text-left transition-all duration-150 ${
                selectedMode === m.key ? m.color : "border-border hover:border-muted-foreground/40"
              }`}
            >
              {selectedMode === m.key && (
                <div className="absolute top-3 right-3">
                  <Check className="h-5 w-5 text-primary" />
                </div>
              )}
              <div className={`mb-3 ${selectedMode === m.key ? "text-foreground" : "text-muted-foreground"}`}>
                {m.icon}
              </div>
              <div className="font-semibold text-base">{m.label}</div>
              <p className="text-xs text-muted-foreground mt-1">{m.desc}</p>
            </button>
          ))}
        </div>
      </div>

      {/* ─── Exchange/Broker-specific settings ─── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            {config.asset_class === "equities" ? "Broker Settings" : "Exchange Settings"}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          {config.asset_class === "equities" ? (
            <>
              <FormField label="Broker">
                <select
                  value={config.broker}
                  onChange={(e) => update({ broker: e.target.value })}
                  className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                >
                  <option value="alpaca">Alpaca</option>
                </select>
              </FormField>

              <FormField label="Universe Mode">
                <select
                  value={config.stock_universe_mode}
                  onChange={(e) => update({ stock_universe_mode: e.target.value })}
                  className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                >
                  <option value="manual">Manual Tickers</option>
                  <option value="filtered">Filtered (Dynamic)</option>
                </select>
              </FormField>

              {config.stock_universe_mode === "manual" && (
                <FormField label="Tickers" hint="Comma-separated symbols">
                  <input
                    type="text"
                    value={config.stock_tickers}
                    onChange={(e) => update({ stock_tickers: e.target.value })}
                    placeholder="AAPL, MSFT, NVDA, TSLA"
                    className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </FormField>
              )}

              {config.stock_universe_mode === "filtered" && (
                <FormField label="Sector Include" hint="Comma-separated, leave blank for all">
                  <input
                    type="text"
                    value={config.stock_sector_include}
                    onChange={(e) => update({ stock_sector_include: e.target.value })}
                    placeholder="Technology, Healthcare"
                    className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </FormField>
              )}

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                <FormField label="Min Price ($)">
                  <input
                    type="number"
                    value={config.stock_min_price}
                    onChange={(e) => update({ stock_min_price: +e.target.value })}
                    className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </FormField>
                <FormField label="Max Price ($)">
                  <input
                    type="number"
                    value={config.stock_max_price}
                    onChange={(e) => update({ stock_max_price: +e.target.value })}
                    className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </FormField>
                <FormField label="Min Volume">
                  <input
                    type="number"
                    value={config.stock_min_volume}
                    onChange={(e) => update({ stock_min_volume: +e.target.value })}
                    className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </FormField>
              </div>

              <ToggleRow
                label="Allow extended hours trading"
                checked={config.allow_extended_hours}
                onChange={(v) => update({ allow_extended_hours: v })}
              />
            </>
          ) : (
            <>
              <FormField label="Exchange">
                <select
                  value={config.exchange}
                  onChange={(e) => update({ exchange: e.target.value })}
                  className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                >
                  <option value="polymarket">Polymarket</option>
                  <option value="kalshi">Kalshi</option>
                </select>
              </FormField>

              <div className="grid grid-cols-2 gap-4">
                <FormField label="Max Tracked Markets">
                  <input
                    type="number"
                    value={config.max_tracked_markets}
                    onChange={(e) => update({ max_tracked_markets: +e.target.value })}
                    className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </FormField>
                <FormField label="Max Subscribed Markets">
                  <input
                    type="number"
                    value={config.max_subscribed_markets}
                    onChange={(e) => update({ max_subscribed_markets: +e.target.value })}
                    className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                  />
                </FormField>
              </div>

              <FormField label="Include Categories" hint="Comma-separated, blank for all">
                <input
                  type="text"
                  value={config.include_categories}
                  onChange={(e) => update({ include_categories: e.target.value })}
                  placeholder="politics, sports, crypto"
                  className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </FormField>

              <FormField label="Exclude Categories" hint="Comma-separated">
                <input
                  type="text"
                  value={config.exclude_categories}
                  onChange={(e) => update({ exclude_categories: e.target.value })}
                  placeholder=""
                  className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </FormField>

              <FormField label="Market Slugs" hint="Comma-separated slugs to target specific markets">
                <input
                  type="text"
                  value={config.market_slugs.join(", ")}
                  onChange={(e) =>
                    update({ market_slugs: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })
                  }
                  placeholder="will-trump-win, bitcoin-100k"
                  className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </FormField>
            </>
          )}
        </CardContent>
      </Card>

      {/* ─── Strategies ─── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="h-5 w-5" />
            Strategies
          </CardTitle>
          <CardDescription>
            Select which strategies run for {config.asset_class === "equities" ? "stocks" : "prediction markets"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {filteredStrategies.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4 text-center">
              No strategies available for this mode
            </p>
          ) : (
            <div className="grid gap-2 sm:grid-cols-2">
              {filteredStrategies.map((s) => {
                const active = config.strategies.includes(s.name);
                return (
                  <button
                    key={s.name}
                    onClick={() => {
                      if (active) {
                        update({ strategies: config.strategies.filter((n) => n !== s.name) });
                      } else {
                        update({ strategies: [...config.strategies, s.name] });
                      }
                    }}
                    className={`flex items-start gap-3 p-3.5 rounded-lg border text-left transition-all ${
                      active ? "border-primary/50 bg-accent/40" : "border-border hover:border-muted-foreground/40"
                    }`}
                  >
                    <div className={`mt-0.5 h-4 w-4 rounded border flex items-center justify-center shrink-0 ${
                      active ? "bg-primary border-primary" : "border-muted-foreground/40"
                    }`}>
                      {active && <Check className="h-3 w-3 text-primary-foreground" />}
                    </div>
                    <div className="min-w-0">
                      <div className="text-sm font-medium">{s.name.replaceAll("_", " ")}</div>
                      <div className="text-xs text-muted-foreground mt-0.5">{s.description}</div>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {/* ─── Intelligence / NLP ─── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Intelligence & NLP
          </CardTitle>
          <CardDescription>Configure news ingestion and LLM-powered analysis</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid grid-cols-2 gap-4">
            <FormField label="NLP Provider">
              <select
                value={config.nlp_provider}
                onChange={(e) => update({ nlp_provider: e.target.value })}
                className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              >
                <option value="mock">Mock (No real data)</option>
                <option value="newsapi">NewsAPI</option>
                <option value="polygon">Polygon.io</option>
              </select>
            </FormField>
            <FormField label="LLM Provider">
              <select
                value={config.llm_provider}
                onChange={(e) => update({ llm_provider: e.target.value })}
                className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              >
                <option value="none">None</option>
                <option value="openai">OpenAI (GPT-4o-mini)</option>
              </select>
            </FormField>
          </div>
          <p className="text-xs text-muted-foreground">
            API keys for providers must be set in the environment (.env). The GUI does not handle secrets.
          </p>
        </CardContent>
      </Card>

      {/* ─── Decision Engine ─── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Decision Engine
          </CardTitle>
          <CardDescription>Tune how the ensemble of layers produces trade signals</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <FormField label="Mode">
            <div className="flex gap-2">
              {(["conservative", "balanced", "aggressive"] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => update({ decision_mode: mode })}
                  className={`flex-1 py-2 rounded-md border text-sm font-medium capitalize transition-colors ${
                    config.decision_mode === mode
                      ? mode === "aggressive"
                        ? "border-red-500/50 bg-red-950/20 text-red-300"
                        : mode === "balanced"
                        ? "border-amber-500/50 bg-amber-950/20 text-amber-300"
                        : "border-emerald-500/50 bg-emerald-950/20 text-emerald-300"
                      : "border-border hover:border-muted-foreground/40"
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
          </FormField>

          <SliderField
            label="L1 (Rules) Weight"
            value={config.ensemble_weight_l1}
            onChange={(v) => update({ ensemble_weight_l1: v })}
          />
          <SliderField
            label="L2 (ML) Weight"
            value={config.ensemble_weight_l2}
            onChange={(v) => update({ ensemble_weight_l2: v })}
          />
          <SliderField
            label="L3 (NLP) Weight"
            value={config.ensemble_weight_l3}
            onChange={(v) => update({ ensemble_weight_l3: v })}
          />
          <SliderField
            label="Min Ensemble Confidence"
            value={config.min_ensemble_confidence}
            onChange={(v) => update({ min_ensemble_confidence: v })}
          />

          <div className="grid grid-cols-2 gap-4">
            <FormField label="Min Layers Agree">
              <input
                type="number"
                min={1}
                max={3}
                value={config.min_layers_agree}
                onChange={(e) => update({ min_layers_agree: +e.target.value })}
                className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </FormField>
            <FormField label="Min Evidence Signals">
              <input
                type="number"
                min={1}
                value={config.min_evidence_signals}
                onChange={(e) => update({ min_evidence_signals: +e.target.value })}
                className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </FormField>
          </div>
        </CardContent>
      </Card>

      {/* ─── Risk Limits ─── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Risk Limits
          </CardTitle>
          <CardDescription>
            {config.asset_class === "equities" ? "Dollar-based limits for stock trading" : "Position and exposure limits"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {config.asset_class === "equities" ? (
            <div className="grid grid-cols-2 gap-4">
              <FormField label="Max Per Position ($)">
                <input type="number" value={config.stock_max_position_dollars}
                  onChange={(e) => update({ stock_max_position_dollars: +e.target.value })}
                  className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring" />
              </FormField>
              <FormField label="Max Portfolio ($)">
                <input type="number" value={config.stock_max_portfolio_dollars}
                  onChange={(e) => update({ stock_max_portfolio_dollars: +e.target.value })}
                  className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring" />
              </FormField>
              <FormField label="Max Daily Loss ($)">
                <input type="number" value={config.stock_max_daily_loss_dollars}
                  onChange={(e) => update({ stock_max_daily_loss_dollars: +e.target.value })}
                  className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring" />
              </FormField>
              <FormField label="Max Open Positions">
                <input type="number" value={config.stock_max_open_positions}
                  onChange={(e) => update({ stock_max_open_positions: +e.target.value })}
                  className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring" />
              </FormField>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              <FormField label="Max Per Market ($)">
                <input type="number" value={config.max_position_per_market}
                  onChange={(e) => update({ max_position_per_market: +e.target.value })}
                  className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring" />
              </FormField>
              <FormField label="Max Total Exposure ($)">
                <input type="number" value={config.max_total_exposure}
                  onChange={(e) => update({ max_total_exposure: +e.target.value })}
                  className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring" />
              </FormField>
              <FormField label="Max Daily Loss ($)">
                <input type="number" value={config.max_daily_loss}
                  onChange={(e) => update({ max_daily_loss: +e.target.value })}
                  className="w-full bg-input border rounded-md px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring" />
              </FormField>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ─── Trading Mode ─── */}
      <Card className={isLive ? "border-red-700/50" : ""}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className={`h-5 w-5 ${isLive ? "text-red-400" : ""}`} />
            Trading Mode
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid grid-cols-2 gap-4">
            <button
              onClick={() => update({ dry_run: true, enable_live_trading: false, live_trading_acknowledged: false })}
              className={`p-5 rounded-xl border-2 text-center transition-all ${
                config.dry_run
                  ? "border-emerald-500/60 bg-emerald-950/20"
                  : "border-border hover:border-muted-foreground/40"
              }`}
            >
              <div className="text-lg font-semibold text-emerald-400">Dry Run</div>
              <p className="text-xs text-muted-foreground mt-1">Simulated — no real orders</p>
            </button>
            <button
              onClick={() => update({ dry_run: false })}
              className={`p-5 rounded-xl border-2 text-center transition-all ${
                !config.dry_run
                  ? "border-red-500/60 bg-red-950/20"
                  : "border-border hover:border-muted-foreground/40"
              }`}
            >
              <div className="text-lg font-semibold text-red-400">Live</div>
              <p className="text-xs text-muted-foreground mt-1">Real orders — use with caution</p>
            </button>
          </div>

          {!config.dry_run && (
            <div className="rounded-lg border border-red-700/50 bg-red-950/20 p-5 space-y-4">
              <div className="flex items-center gap-2 text-red-400 font-semibold text-sm">
                <AlertTriangle className="h-4 w-4" />
                Live Trading Safety Gates
              </div>
              <p className="text-xs text-muted-foreground">
                All three gates must be open AND valid credentials must be present in the environment.
              </p>
              <ToggleRow
                label="I want to enable live trading"
                checked={config.enable_live_trading}
                onChange={(v) => update({ enable_live_trading: v })}
                danger
              />
              <ToggleRow
                label="I acknowledge that real money is at risk"
                checked={config.live_trading_acknowledged}
                onChange={(v) => update({ live_trading_acknowledged: v })}
                danger
              />
            </div>
          )}
        </CardContent>
      </Card>

      {/* ─── Validation feedback ─── */}
      {errors.length > 0 && (
        <div className="rounded-lg border border-red-700/50 bg-red-950/30 p-4">
          <div className="font-semibold text-red-400 text-sm mb-2">Validation Errors</div>
          <ul className="text-sm text-red-300 space-y-1">
            {errors.map((e, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-red-500 mt-0.5">×</span> {e}
              </li>
            ))}
          </ul>
        </div>
      )}
      {warnings.length > 0 && (
        <div className="rounded-lg border border-amber-700/50 bg-amber-950/20 p-4">
          <ul className="text-sm text-amber-300 space-y-1">
            {warnings.map((w, i) => (
              <li key={i} className="flex items-start gap-2">
                <AlertTriangle className="h-3.5 w-3.5 text-amber-500 mt-0.5 shrink-0" /> {w}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ─── Actions ─── */}
      <div className="flex flex-col sm:flex-row gap-3 pb-8">
        <Button variant="outline" onClick={handleValidate} className="sm:w-auto">
          {validated ? <Check className="h-4 w-4 mr-1.5 text-emerald-400" /> : null}
          {validated ? "Config Valid" : "Validate"}
        </Button>
        <Button
          variant={isLive ? "destructive" : "success"}
          onClick={handleStart}
          disabled={loading}
          className="sm:w-auto"
        >
          {loading ? "Starting..." : isLive ? "⚠ Start Live Trading" : "Start Dry Run"}
        </Button>
      </div>
    </div>
  );
}

/* ── Small helper components ─────────────────────────────────── */

function FormField({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-sm font-medium mb-1.5">{label}</label>
      {children}
      {hint && <p className="text-[11px] text-muted-foreground mt-1">{hint}</p>}
    </div>
  );
}

function SliderField({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <label className="text-sm font-medium">{label}</label>
        <span className="text-sm font-mono tabular-nums text-muted-foreground">{(value * 100).toFixed(0)}%</span>
      </div>
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={value}
        onChange={(e) => onChange(+e.target.value)}
        className="w-full"
      />
    </div>
  );
}

function ToggleRow({
  label,
  checked,
  onChange,
  danger,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  danger?: boolean;
}) {
  return (
    <label className="flex items-center gap-3 cursor-pointer group">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative inline-flex h-6 w-11 shrink-0 rounded-full border-2 border-transparent transition-colors ${
          checked
            ? danger
              ? "bg-red-600"
              : "bg-emerald-600"
            : "bg-secondary"
        }`}
      >
        <span
          className={`pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow-sm transition-transform ${
            checked ? "translate-x-5" : "translate-x-0"
          }`}
        />
      </button>
      <span className="text-sm">{label}</span>
    </label>
  );
}
