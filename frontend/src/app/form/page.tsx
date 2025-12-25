'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import Image from 'next/image';
import { Home } from 'lucide-react';

export default function FormPage() {
    const router = useRouter();
    const [loading, setLoading] = useState(false);

    const [formData, setFormData] = useState({
        Age: '' as any,
        Sex: '' as any,
        height: '' as any,
        weight: '' as any,
        HighBP: 0,
        HighChol: 0,
        CholCheck: 0,
        Smoker: 0,
        Stroke: 0,
        HeartDiseaseorAttack: 0,
        PhysActivity: 0,
        Fruits: 0,
        Veggies: 0,
        HvyAlcoholConsump: 0,
        AnyHealthcare: 0,
        DiffWalk: 0,
    });

    const calculateBMI = () => {
        const heightM = formData.height / 100;
        const val = formData.weight / (heightM * heightM);
        return isNaN(val) ? 0 : val;
    };

    const bmi = calculateBMI();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);

        // Sanitize data -> ensure numbers are valid
        const payload = {
            ...formData,
            Age: Number(formData.Age) || 30,
            height: Number(formData.height) || 170,
            weight: Number(formData.weight) || 70,
            Sex: formData.Sex === '' ? 1 : Number(formData.Sex),
            BMI: bmi || 24.2,
            Education: 4.0, // Keeping these as you had them
            Income: 5.0,
        };

        try {
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
            const response = await fetch(`${apiUrl}/predict/all`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (response.ok) {
                const results = await response.json();
                localStorage.setItem('healthResults', JSON.stringify(results));
                router.push('/dashboard');
            } else {
                alert('Error analyzing data. Please try again.');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Could not connect to API. Make sure the backend is running.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-[#080c14] text-white selection:bg-blue-500/30 font-sans relative">
            {/* --- Background (No Grid) --- */}
            <div className="fixed inset-0 z-0 pointer-events-none">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-blue-900/10 via-[#080c14] to-[#080c14]"></div>
            </div>

            {/* --- Navigation --- */}
            <nav className="fixed top-0 w-full z-50 border-b border-white/5 bg-[#080c14]/80 backdrop-blur-xl">
                <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <Link href="/" className="flex items-center gap-3 group">
                            <Image src="/logo.png" alt="HealthScope Logo" width={28} height={28} className="group-hover:scale-110 transition-transform" />
                            <span className="text-xl font-bold tracking-tight text-white cursor-pointer hover:text-blue-400 transition-colors">HealthScope</span>
                        </Link>
                    </div>
                    <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-300">
                        <Link href="/" className="hover:text-white transition-colors cursor-pointer hover:underline flex items-center gap-2">
                            <Home size={14} />
                            Back to Home
                        </Link>
                    </div>
                </div>
            </nav>

            <main className="relative z-10 pt-32 pb-20 container mx-auto px-6 max-w-5xl">
                {/* Header */}
                <div className="text-center mb-16">
                    <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 text-xs font-semibold tracking-wide uppercase mb-6">
                        Secure Intake Portal
                    </div>
                    <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-4 text-blue-200">
                        Intake Form
                    </h1>
                    <p className="text-slate-400 max-w-2xl mx-auto">
                        Your data is processed locally within our encrypted neural network for instant risk profiling.
                    </p>
                </div>

                <form onSubmit={handleSubmit} className="space-y-8">

                    {/* Section 1: Biometrics */}
                    <div className="group relative p-8 rounded-3xl bg-[#0e1629]/80 backdrop-blur-sm border border-white/10 hover:border-cyan-500/30 transition-all duration-300">
                        <div className="flex items-center gap-4 mb-8 pb-4 border-b border-white/5">
                            <div className="w-10 h-10 rounded-full bg-cyan-500/10 flex items-center justify-center text-cyan-400 font-bold border border-cyan-500/20">1</div>
                            <h2 className="text-xl font-bold text-white">Biometrics & Vitals</h2>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Age</label>
                                <input required type="number" value={formData.Age} onChange={(e) => setFormData({ ...formData, Age: e.target.value === '' ? '' : parseInt(e.target.value) })}
                                    className="w-full bg-[#080c14] border border-white/10 rounded-xl px-4 py-3 text-white focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none transition-all placeholder-slate-600 appearance-none [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none" />
                            </div>
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Gender</label>
                                <div className="relative">
                                    <select required value={formData.Sex} onChange={(e) => setFormData({ ...formData, Sex: parseInt(e.target.value) })}
                                        className="w-full bg-[#080c14] border border-white/10 rounded-xl px-4 py-3 text-white focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none transition-all appearance-none cursor-pointer">
                                        <option value="" disabled>Select Gender</option>
                                        <option value={1}>Male</option>
                                        <option value={0}>Female</option>
                                    </select>
                                    <div className="absolute inset-y-0 right-0 flex items-center px-4 pointer-events-none text-slate-500">
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                                    </div>
                                </div>
                            </div>
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Height (cm)</label>
                                <input required type="number" value={formData.height} onChange={(e) => setFormData({ ...formData, height: e.target.value === '' ? '' : parseInt(e.target.value) })}
                                    className="w-full bg-[#080c14] border border-white/10 rounded-xl px-4 py-3 text-white focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none transition-all appearance-none [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none" />
                            </div>
                            <div className="space-y-2">
                                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Weight (kg)</label>
                                <input required type="number" value={formData.weight} onChange={(e) => setFormData({ ...formData, weight: e.target.value === '' ? '' : parseInt(e.target.value) })}
                                    className="w-full bg-[#080c14] border border-white/10 rounded-xl px-4 py-3 text-white focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none transition-all appearance-none [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none" />
                            </div>
                        </div>

                        {/* Live BMI Indicator */}
                        <div className="mt-8 flex items-center justify-end gap-4 p-3 rounded-lg bg-cyan-500/5 border border-cyan-500/10 w-fit ml-auto">
                            <div className="text-xs font-medium text-cyan-200/70 uppercase tracking-widest">
                                Calculated BMI Index
                            </div>
                            <div className="text-xl font-bold text-cyan-400 font-mono">
                                {bmi.toFixed(1)}
                            </div>
                        </div>
                    </div>

                    {/* Section 2: Clinical History */}
                    <div className="group relative p-8 rounded-3xl bg-[#0e1629]/80 backdrop-blur-sm border border-white/10 hover:border-cyan-500/30 transition-all duration-300">
                        <div className="flex items-center gap-4 mb-8 pb-4 border-b border-white/5">
                            <div className="w-10 h-10 rounded-full bg-cyan-500/10 flex items-center justify-center text-cyan-400 font-bold border border-cyan-500/20">2</div>
                            <h2 className="text-xl font-bold text-white">Clinical History</h2>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
                            {[
                                { label: 'Diagnosed High BP', key: 'HighBP' },
                                { label: 'High Cholesterol', key: 'HighChol' },
                                { label: 'Cholesterol Check (5yrs)', key: 'CholCheck' },
                                { label: 'History of Stroke', key: 'Stroke' },
                                { label: 'Heart Disease / Attack', key: 'HeartDiseaseorAttack' },
                                { label: 'Difficulty Walking', key: 'DiffWalk' },
                            ].map((item) => (
                                <div key={item.key} className="flex items-center justify-between p-3 rounded-lg hover:bg-white/5 transition-colors">
                                    <label className="text-sm font-medium text-slate-300">{item.label}</label>
                                    <div className="flex bg-[#080c14] rounded-lg p-1 border border-white/10">
                                        <button type="button" onClick={() => setFormData({ ...formData, [item.key]: 0 })}
                                            className={`px-4 py-1.5 text-xs font-bold rounded-md transition-all cursor-pointer ${formData[item.key as keyof typeof formData] === 0 ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}>NO</button>
                                        <button type="button" onClick={() => setFormData({ ...formData, [item.key]: 1 })}
                                            className={`px-4 py-1.5 text-xs font-bold rounded-md transition-all cursor-pointer ${formData[item.key as keyof typeof formData] === 1 ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}>YES</button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Section 3: Lifestyle */}
                    <div className="group relative p-8 rounded-3xl bg-[#0e1629]/80 backdrop-blur-sm border border-white/10 hover:border-cyan-500/30 transition-all duration-300">
                        <div className="flex items-center gap-4 mb-8 pb-4 border-b border-white/5">
                            <div className="w-10 h-10 rounded-full bg-cyan-500/10 flex items-center justify-center text-cyan-400 font-bold border border-cyan-500/20">3</div>
                            <h2 className="text-xl font-bold text-white">Lifestyle Factors</h2>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
                            {[
                                { label: 'Smoker (Lifetime)', key: 'Smoker' },
                                { label: 'Heavy Alcohol', key: 'HvyAlcoholConsump' },
                                { label: 'Physical Activity', key: 'PhysActivity' },
                                { label: 'Daily Fruits', key: 'Fruits' },
                                { label: 'Daily Veggies', key: 'Veggies' },
                                { label: 'Healthcare Coverage', key: 'AnyHealthcare' },
                            ].map((item) => (
                                <div key={item.key} className="flex items-center justify-between p-3 rounded-lg hover:bg-white/5 transition-colors">
                                    <label className="text-sm font-medium text-slate-300 uppercase tracking-wide">{item.label}</label>
                                    <div className="flex bg-[#080c14] rounded-lg p-1 border border-white/10">
                                        <button type="button" onClick={() => setFormData({ ...formData, [item.key]: 0 })}
                                            className={`px-4 py-1.5 text-xs font-bold rounded-md transition-all cursor-pointer ${formData[item.key as keyof typeof formData] === 0 ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}>NO</button>
                                        <button type="button" onClick={() => setFormData({ ...formData, [item.key]: 1 })}
                                            className={`px-4 py-1.5 text-xs font-bold rounded-md transition-all cursor-pointer ${formData[item.key as keyof typeof formData] === 1 ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-300'}`}>YES</button>
                                    </div>
                                </div>
                            ))}
                        </div>


                    </div>

                    {/* Action */}
                    <div className="flex justify-end pt-8 pb-20">
                        <button
                            type="submit"
                            disabled={loading}
                            className="bg-blue-300 hover:bg-blue-300 text-black text-lg font-bold py-4 px-12 rounded-2xl  hover:shadow-[0_0_50px_-10px_rgba(59,130,246,0.6)] transform hover:scale-[1.02] transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-3 cursor-pointer"
                        >
                            {loading ? (
                                <>
                                    <svg className="animate-spin h-5 w-5 text-black" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                    Analyzing Health Data...
                                </>
                            ) : (
                                <>
                                    Run Analysis
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M13 7l5 5m0 0l-5 5m5-5H6"></path></svg>
                                </>
                            )}
                        </button>
                    </div>
                </form>
            </main>
        </div>
    );
}
