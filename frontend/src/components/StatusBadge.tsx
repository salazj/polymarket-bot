import { Badge } from "@/components/ui/badge";
import { Circle } from "lucide-react";

interface Props {
  running: boolean;
  status: string;
  mode: string;
}

export function StatusBadge({ running, status, mode }: Props) {
  if (status === "error") {
    return (
      <Badge variant="destructive" className="gap-1.5">
        <Circle className="h-2 w-2 fill-current" /> Error
      </Badge>
    );
  }
  if (running) {
    return (
      <Badge variant={mode === "live" ? "warning" : "success"} className="gap-1.5">
        <Circle className="h-2 w-2 fill-current animate-pulse" />
        {mode === "live" ? "Live Trading" : "Dry Run"}
      </Badge>
    );
  }
  return (
    <Badge variant="secondary" className="gap-1.5">
      <Circle className="h-2 w-2 fill-current opacity-40" /> Stopped
    </Badge>
  );
}
