import { AlertTriangle } from "lucide-react";

export function LiveBanner({ active }: { active: boolean }) {
  if (!active) return null;
  return (
    <div className="bg-red-700 text-white px-4 py-2.5 text-center text-sm font-semibold flex items-center justify-center gap-2 shadow-lg shadow-red-900/20">
      <AlertTriangle className="h-4 w-4 animate-pulse" />
      LIVE TRADING ENABLED — Real orders will be submitted
      <AlertTriangle className="h-4 w-4 animate-pulse" />
    </div>
  );
}
