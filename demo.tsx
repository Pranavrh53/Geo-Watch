import { KnotAnimation } from "@/components/ui/knot-animation";

const panelBase =
  "relative overflow-hidden rounded-none border-4 bg-white shadow-[6px_6px_0_rgba(0,184,169,0.2)]";

const DemoOne = () => {
  return (
    <div className="flex min-h-screen w-full items-center justify-center bg-[#f5f5f5] p-6">
      <div className="relative w-full max-w-6xl border-[5px] border-[#00B8A9] bg-[#f7fffd] p-6 shadow-[10px_10px_0_rgba(0,184,169,0.2)]">
        <div className="mb-5 border-b-4 border-[#B2E600] pb-4">
          <h1 className="text-xl font-bold tracking-wide text-[#1a1a1a]">Geo-Watch Analysis Terminal</h1>
          <p className="mt-2 text-sm text-[#45505a]">
            Loading before and after imagery, then preparing feature analysis insights.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
          <section className={`${panelBase} border-[#5555FF] p-4 lg:col-span-2`}>
            <div className="mb-4 flex items-center justify-between border-b-2 border-[#d9e2ff] pb-3">
              <h2 className="text-sm font-bold uppercase tracking-wider text-[#3333CC]">
                Before / After Images
              </h2>
              <span className="rounded-sm border-2 border-[#5555FF] bg-[#eef0ff] px-2 py-1 text-[11px] font-bold text-[#3333CC]">
                Analyzing...
              </span>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="border-4 border-[#B2E600] bg-[#f9ffef] p-3">
                <div className="mb-2 text-xs font-bold uppercase tracking-wide text-[#6a8000]">Before</div>
                <div className="flex h-44 items-center justify-center border-2 border-dashed border-[#B2E600]/70 bg-white">
                  <KnotAnimation width={48} height={22} speedA={0.038} speedB={0.02} />
                </div>
              </div>

              <div className="border-4 border-[#FF7070] bg-[#fff3f3] p-3">
                <div className="mb-2 text-xs font-bold uppercase tracking-wide text-[#b73838]">After</div>
                <div className="flex h-44 items-center justify-center border-2 border-dashed border-[#FF7070]/70 bg-white">
                  <KnotAnimation width={48} height={22} speedA={0.04} speedB={0.024} />
                </div>
              </div>
            </div>
          </section>

          <section className={`${panelBase} border-[#00B8A9] p-4`}>
            <div className="mb-4 flex items-center justify-between border-b-2 border-[#bdebe6] pb-3">
              <h2 className="text-sm font-bold uppercase tracking-wider text-[#007a70]">Feature Analysis</h2>
              <span className="h-2.5 w-2.5 animate-pulse rounded-sm bg-[#00B8A9]" />
            </div>

            <div className="mb-4 flex h-48 items-center justify-center border-2 border-dashed border-[#00B8A9]/70 bg-[#fbfffe]">
              <KnotAnimation width={44} height={21} speedA={0.045} speedB={0.028} />
            </div>

            <ul className="space-y-2 text-xs text-[#33414d]">
              <li className="flex items-center gap-2">
                <span className="h-2 w-2 animate-pulse rounded-sm bg-[#B2E600]" />
                Aligning multispectral tiles
              </li>
              <li className="flex items-center gap-2">
                <span className="h-2 w-2 animate-pulse rounded-sm bg-[#5555FF]" />
                Computing temporal change vectors
              </li>
              <li className="flex items-center gap-2">
                <span className="h-2 w-2 animate-pulse rounded-sm bg-[#FF7070]" />
                Scoring feature-level confidence
              </li>
            </ul>
          </section>
        </div>
      </div>
    </div>
  );
};

export { DemoOne };
