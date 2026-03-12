import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { LiveBanner } from "./LiveBanner";
import { useBotStatus } from "@/hooks/useBotStatus";

export function Layout() {
  const { botStatus } = useBotStatus();

  return (
    <div className="min-h-screen">
      <LiveBanner active={botStatus.live_trading} />
      <Sidebar />
      <main className="md:pl-60 pb-20 md:pb-0">
        <div className="mx-auto max-w-7xl p-4 sm:p-6 lg:p-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
