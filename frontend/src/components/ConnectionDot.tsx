import { cn } from "@/lib/utils";

export function ConnectionDot({ connected }: { connected: boolean }) {
  return (
    <span className="relative flex h-2.5 w-2.5" title={connected ? "Connected" : "Disconnected"}>
      {connected && (
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
      )}
      <span
        className={cn(
          "relative inline-flex h-2.5 w-2.5 rounded-full",
          connected ? "bg-emerald-500" : "bg-red-500"
        )}
      />
    </span>
  );
}
