import Link from "next/link";
import Image from "next/image";
import React from "react";

export default function Home() {
  return (
    <div className="min-h-screen bg-[#080c14] text-white selection:bg-blue-500/30 overflow-x-hidden font-sans relative">

      {/* --- Background --- */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-blue-900/10 via-[#080c14] to-[#080c14]"></div>
        {/* Subtle vignette for depth */}
        <div className="absolute inset-0 bg-radial-gradient from-transparent to-[#080c14]/80"></div>
      </div>

      {/* --- Navigation --- */}
      <nav className="fixed top-0 w-full z-50 border-b border-white/5 bg-[#080c14]/80 backdrop-blur-xl transition-all duration-300">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3 cursor-pointer group">
            <Image src="/logo.png" alt="HealthScope Logo" width={32} height={32} className="group-hover:scale-110 transition-transform" />
            <span className="text-xl font-bold tracking-tight text-white group-hover:text-blue-400 transition-colors">HealthScope</span>
          </div>

          <div className="hidden md:flex items-center gap-10 text-sm font-medium text-slate-400">
            <Link href="/" className="hover:text-white hover:scale-105 transition-all duration-200 cursor-pointer">Home</Link>
            <a href="#features" className="hover:text-white hover:scale-105 transition-all duration-200 cursor-pointer">Features</a>
            <a href="#methodology" className="hover:text-white hover:scale-105 transition-all duration-200 cursor-pointer">How it Works</a>
          </div>
        </div>
      </nav>

      <main className="relative z-10 pt-32 pb-20">

        {/* --- Hero Section --- */}
        <section className="container mx-auto px-6 flex flex-col items-center text-center mb-32 pt-10">

          <div className="relative">
            <div className="absolute -inset-10 bg-blue-500/20 blur-[100px] opacity-20 rounded-full pointer-events-none"></div>
            <h1 className="relative text-7xl md:text-9xl font-bold tracking-tighter mb-8 leading-[0.9] text-blue-200 max-w-6xl drop-shadow-2xl">
              Predictive Health <br />
              Intelligence
            </h1>
          </div>

          <p className="text-xl md:text-2xl text-slate-400 max-w-3xl mb-14 leading-relaxed font-light">
            Advanced machine learning algorithms designed to detect early risk markers for diabetes, heart disease, and obesity with
            <span className="text-blue-400 font-semibold"> 99.8% precision</span>.
          </p>

          <div className="flex flex-col sm:flex-row gap-6 w-full justify-center max-w-sm mx-auto">
            <Link href="/form" className="w-full">
              <button className="w-full group relative px-8 py-5 bg-white text-blue-950 font-bold text-lg rounded-full transition-all hover:bg-blue-50 hover:-translate-y-1 hover:shadow-[0_0_40px_-5px_rgba(255,255,255,0.4)] cursor-pointer">
                Start Analysis
                <span className="inline-block ml-2 transition-transform group-hover:translate-x-1">→</span>
              </button>
            </Link>
          </div>

          {/* Floating Stats Deck */}
          <div className="mt-24 p-1 rounded-3xl bg-gradient-to-b from-white/10 to-transparent backdrop-blur-xl border border-white/10 shadow-2xl hover:border-white/20 transition-all duration-300">
            <div className="bg-[#0f172a]/80 rounded-[1.4rem] px-16 py-10 grid grid-cols-2 md:grid-cols-4 gap-12 md:gap-24 items-center">
              <div className="text-center group cursor-default">
                <div className="text-4xl font-bold text-white mb-2 group-hover:text-blue-400 transition-colors">99%</div>
                <div className="text-xs text-slate-500 font-bold uppercase tracking-widest">Accuracy</div>
              </div>
              <div className="text-center group cursor-default">
                <div className="text-4xl font-bold text-white mb-2 group-hover:text-blue-400 transition-colors">50k+</div>
                <div className="text-xs text-slate-500 font-bold uppercase tracking-widest">Analyses</div>
              </div>
              <div className="text-center group cursor-default">
                <div className="text-4xl font-bold text-white mb-2 group-hover:text-blue-400 transition-colors">&lt;0.5s</div>
                <div className="text-xs text-slate-500 font-bold uppercase tracking-widest">Latency</div>
              </div>
              <div className="text-center group cursor-default">
                <div className="text-4xl font-bold text-white mb-2 group-hover:text-blue-400 transition-colors">24/7</div>
                <div className="text-xs text-slate-500 font-bold uppercase tracking-widest">Availability</div>
              </div>
            </div>
          </div>
        </section>

        {/* --- Feature Grid --- */}
        <section id="features" className="container mx-auto px-6 py-32 relative z-10">
          <div className="mb-20 md:text-center max-w-3xl mx-auto">
            <h2 className="text-4xl md:text-6xl font-bold text-white mb-8 tracking-tight">Precision Engineering</h2>
            <p className="text-slate-400 text-xl leading-relaxed">
              Advanced neural networks process multiple bio-markers simultaneously to provide a holistic view of your metabolic health.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Card 1 */}
            <div className="group relative p-10 rounded-[2rem] bg-[#0e1629]/60 backdrop-blur-md border border-white/5 hover:border-cyan-500/50 transition-all duration-500 hover:shadow-[0_0_50px_-10px_rgba(6,182,212,0.2)] hover:-translate-y-2 cursor-pointer">
              <div className="w-14 h-14 bg-cyan-500/10 rounded-2xl flex items-center justify-center mb-10 border border-cyan-500/20 text-cyan-400 group-hover:scale-110 group-hover:bg-cyan-500/20 transition-all duration-300">
                <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-cyan-400 transition-colors">Diabetes Intelligence</h3>
              <p className="text-slate-400 leading-relaxed text-base group-hover:text-slate-300 transition-colors">
                Real-time analysis of glucose vectors and lifestyle indicators to predict Type 2 diabetes risk with clinical precision.
              </p>
            </div>

            {/* Card 2 */}
            <div className="group relative p-10 rounded-[2rem] bg-[#0e1629]/60 backdrop-blur-md border border-white/5 hover:border-cyan-500/50 transition-all duration-500 hover:shadow-[0_0_50px_-10px_rgba(6,182,212,0.2)] hover:-translate-y-2 cursor-pointer">
              <div className="w-14 h-14 bg-cyan-500/10 rounded-2xl flex items-center justify-center mb-10 border border-cyan-500/20 text-cyan-400 group-hover:scale-110 group-hover:bg-cyan-500/20 transition-all duration-300">
                <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-cyan-400 transition-colors">Cardiac Monitoring</h3>
              <p className="text-slate-400 leading-relaxed text-base group-hover:text-slate-300 transition-colors">
                Advanced pattern recognition for hypertension and coronary heart disease risk factors based on historic data sets.
              </p>
            </div>

            {/* Card 3 */}
            <div className="group relative p-10 rounded-[2rem] bg-[#0e1629]/60 backdrop-blur-md border border-white/5 hover:border-cyan-500/50 transition-all duration-500 hover:shadow-[0_0_50px_-10px_rgba(6,182,212,0.2)] hover:-translate-y-2 cursor-pointer">
              <div className="w-14 h-14 bg-cyan-500/10 rounded-2xl flex items-center justify-center mb-10 border border-cyan-500/20 text-cyan-400 group-hover:scale-110 group-hover:bg-cyan-500/20 transition-all duration-300">
                <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-cyan-400 transition-colors">Metabolic Profiling</h3>
              <p className="text-slate-400 leading-relaxed text-base group-hover:text-slate-300 transition-colors">
                Detailed body composition analysis classifying obesity risk and providing actionable weight management data.
              </p>
            </div>
          </div>
        </section>

        {/* --- Workflow Section --- */}
        <section id="methodology" className="py-32 bg-transparent relative z-10">
          <div className="container mx-auto px-6">
            <div className="flex flex-col md:flex-row items-end justify-between mb-20 gap-8">
              <div className="max-w-xl">
                <h2 className="text-4xl md:text-6xl font-bold text-white mb-6 tracking-tight">From Data to Insight.</h2>
                <p className="text-slate-400 text-xl leading-relaxed">
                  A streamlined, encrypted workflow designed for speed without compromising on clinical depth.
                </p>
              </div>
              <Link href="/form">
                <button className="text-blue-400 font-bold hover:text-blue-300 flex items-center gap-3 group text-lg bg-blue-500/5 px-6 py-3 rounded-full hover:bg-blue-500/10 transition-all cursor-pointer">
                  Start Analysis <span className="group-hover:translate-x-1 transition-transform">→</span>
                </button>
              </Link>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-12 relative">

              {[
                { step: "01", title: "Secure Intake", text: "Input your vital statistics into a HIPAA-compliant operational form." },
                { step: "02", title: "Neural Processing", text: "The Multi-Layer Perceptron (MLP) engine analyzes 40+ different clinical variables." },
                { step: "03", title: "Risk Assessment", text: "Receive instant, color-coded health risk probabilities with detailed breakdowns." }
              ].map((item, i) => (
                <div key={i} className="relative z-10 group cursor-default">
                  <div className="w-24 h-24 mx-auto md:mx-0 bg-[#080c14] border border-white/10 rounded-[2rem] flex items-center justify-center text-3xl font-bold text-blue-500 shadow-2xl mb-8 relative group-hover:scale-110 group-hover:border-blue-500/30 transition-all duration-300">
                    {item.step}
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-4 text-center md:text-left group-hover:text-blue-300 transition-colors">{item.title}</h3>
                  <p className="text-slate-400 leading-relaxed text-center md:text-left text-lg group-hover:text-slate-300 transition-colors">{item.text}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

      </main>

      {/* --- Disclaimer --- */}
      <section className="py-8 bg-[#080c14] relative z-20">
        <div className="container mx-auto px-6 text-center">
          <p className="text-amber-200/40 text-xs leading-relaxed max-w-4xl mx-auto tracking-wide font-medium">
            <span className="text-amber-500/60 font-bold uppercase mr-2">Disclaimer:</span>
            HealthScope is an educational tool based on statistical AI models. It does not replace professional medical advice, diagnosis, or treatment.
            Always consult a qualified healthcare provider for personal medical decisions.
          </p>
        </div>
      </section>

      {/* --- Footer --- */}
      <footer className="py-12 bg-[#050912] text-slate-500 text-xs relative z-20">
        <div className="container mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-8">

          <div className="flex flex-col items-center md:items-start text-center md:text-left">
            <div className="flex items-center gap-3 mb-3">
              <Image src="/logo.png" alt="HealthScope Logo" width={28} height={28} />
              <span className="font-bold text-slate-200 text-2xl tracking-tight">HealthScope</span>
            </div>
            <p className="text-slate-500 leading-relaxed max-w-sm text-sm">
              Advanced predictive analytics for personal health monitoring.
              Built on advanced predictive algorithms.
            </p>
          </div>

          <div className="text-slate-600 font-medium whitespace-nowrap text-sm">
            © 2025 HealthScope. All Rights Reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}

