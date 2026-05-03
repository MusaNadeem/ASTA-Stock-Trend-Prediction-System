import { SplineSceneBasic } from "@/components/demo";

export default function Home() {
  return (
    <div className="min-h-screen bg-black py-12 px-4">
      <main className="max-w-7xl mx-auto space-y-8">
        <div className="text-center space-y-4 mb-12">
          <h1 className="text-5xl md:text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400">
            Spline 3D Integration
          </h1>
          <p className="text-neutral-400 text-lg max-w-2xl mx-auto">
            Interactive 3D scenes powered by Spline, integrated seamlessly with Next.js and shadcn/ui components.
          </p>
        </div>

        <SplineSceneBasic />

        <div className="grid md:grid-cols-3 gap-6 mt-12">
          <div className="p-6 rounded-lg border border-neutral-800 bg-neutral-900/50">
            <h3 className="text-xl font-semibold text-neutral-50 mb-2">
              Lazy Loading
            </h3>
            <p className="text-neutral-400">
              Components are lazy-loaded for optimal performance, reducing initial bundle size.
            </p>
          </div>
          <div className="p-6 rounded-lg border border-neutral-800 bg-neutral-900/50">
            <h3 className="text-xl font-semibold text-neutral-50 mb-2">
              Fully Typed
            </h3>
            <p className="text-neutral-400">
              Built with TypeScript for type safety and better developer experience.
            </p>
          </div>
          <div className="p-6 rounded-lg border border-neutral-800 bg-neutral-900/50">
            <h3 className="text-xl font-semibold text-neutral-50 mb-2">
              Responsive Design
            </h3>
            <p className="text-neutral-400">
              Works beautifully across all device sizes with Tailwind CSS utilities.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}

