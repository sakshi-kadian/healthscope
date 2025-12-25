'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import Image from 'next/image';
import jsPDF from 'jspdf';
import {
    Activity,
    Heart,
    Scale,
    AlertCircle,
    CheckCircle,
    Download,
    RefreshCw,
    FileText,
    ChevronDown,
    ChevronUp,
    BrainCircuit,
    ArrowRight,
    Home
} from 'lucide-react';

interface HealthResults {
    diabetes_risk: number;
    diabetes_prediction: string;
    heart_risk: number;
    heart_prediction: string;
    obesity_risk: number;
    obesity_level: string;
    bmi: number;
}

export default function DashboardPage() {
    const router = useRouter();
    const [results, setResults] = useState<HealthResults | null>(null);
    const [openSection, setOpenSection] = useState<string | null>(null);

    useEffect(() => {
        const data = localStorage.getItem('healthResults');
        if (data) {
            setResults(JSON.parse(data));
        } else {
            router.push('/form');
        }
    }, [router]);

    if (!results) {
        return (
            <div className="min-h-screen bg-[#080c14] flex items-center justify-center">
                <div className="flex flex-col items-center gap-4">
                    <div className="w-12 h-12 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin"></div>
                    <div className="text-blue-400 font-mono text-sm animate-pulse">Loading Analysis...</div>
                </div>
            </div>
        );
    }

    // Unified Premium Blue Theme for all cards
    const premiumCardStyle = "bg-blue-500/5 border-blue-500/20 hover:border-blue-500/40 text-blue-100 cursor-pointer";
    const premiumIconStyle = "bg-blue-500/10 border-blue-500/20 text-blue-400";
    const premiumBadgeStyle = "bg-blue-500/10 text-blue-400";

    const getRiskLabel = (prob: number) => {
        if (prob < 0.3) return 'Low Risk';
        if (prob < 0.7) return 'Moderate Risk';
        return 'High Risk';
    };

    const overallScore = Math.round(100 - ((results.diabetes_risk + results.heart_risk) / 2) * 100);

    const downloadReport = () => {
        const doc = new jsPDF();

        // Header
        doc.setFontSize(24);
        doc.setTextColor(59, 130, 246); // Blue
        doc.text('HealthScope', 105, 20, { align: 'center' });

        doc.setFontSize(12);
        doc.setTextColor(100, 100, 100);
        doc.text('Clinical Assessment Report', 105, 28, { align: 'center' });

        // Date
        doc.setFontSize(10);
        doc.text(`Generated: ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString()}`, 105, 35, { align: 'center' });

        // Line separator
        doc.setDrawColor(59, 130, 246);
        doc.setLineWidth(0.5);
        doc.line(20, 40, 190, 40);

        // Overall Score
        doc.setFontSize(16);
        doc.setTextColor(0, 0, 0);
        doc.text('Overall Health Score', 20, 50);
        doc.setFontSize(32);
        doc.setTextColor(59, 130, 246);
        doc.text(`${overallScore}/100`, 20, 62);

        // Risk Assessments
        let yPos = 75;

        // Diabetes
        doc.setFontSize(14);
        doc.setTextColor(0, 0, 0);
        doc.text('1. Diabetes Risk Assessment', 20, yPos);
        yPos += 8;
        doc.setFontSize(10);
        doc.setTextColor(60, 60, 60);
        doc.text(`Risk Probability: ${(results.diabetes_risk * 100).toFixed(1)}%`, 25, yPos);
        yPos += 6;
        doc.text(`Assessment: ${getRiskLabel(results.diabetes_risk)}`, 25, yPos);
        yPos += 6;
        doc.text(`Prediction: ${results.diabetes_prediction}`, 25, yPos);
        yPos += 12;

        // Heart Disease
        doc.setFontSize(14);
        doc.setTextColor(0, 0, 0);
        doc.text('2. Cardiac Risk Assessment', 20, yPos);
        yPos += 8;
        doc.setFontSize(10);
        doc.setTextColor(60, 60, 60);
        doc.text(`Risk Probability: ${(results.heart_risk * 100).toFixed(1)}%`, 25, yPos);
        yPos += 6;
        doc.text(`Assessment: ${getRiskLabel(results.heart_risk)}`, 25, yPos);
        yPos += 6;
        doc.text(`Prediction: ${results.heart_prediction}`, 25, yPos);
        yPos += 12;

        // Obesity
        doc.setFontSize(14);
        doc.setTextColor(0, 0, 0);
        doc.text('3. Body Mass Index Profile', 20, yPos);
        yPos += 8;
        doc.setFontSize(10);
        doc.setTextColor(60, 60, 60);
        doc.text(`BMI: ${results.bmi.toFixed(2)}`, 25, yPos);
        yPos += 6;
        doc.text(`Classification: ${results.obesity_level}`, 25, yPos);
        yPos += 15;

        // Recommendations Section
        doc.setFontSize(14);
        doc.setTextColor(0, 0, 0);
        doc.text('Clinical Recommendations', 20, yPos);
        yPos += 8;

        doc.setFontSize(9);
        doc.setTextColor(60, 60, 60);
        const recommendations = [
            '• Monitor blood glucose levels regularly (Fasting & Post-prandial)',
            '• Maintain a low-glycemic index diet rich in fiber',
            '• Regular monitoring of Blood Pressure and Lipid Profile',
            '• Adopt a DASH or Mediterranean diet (low sodium, healthy fats)',
            '• Balance caloric intake with expenditure for weight management'
        ];

        recommendations.forEach(rec => {
            doc.text(rec, 25, yPos);
            yPos += 5;
        });

        // Disclaimer
        yPos += 10;
        doc.setDrawColor(200, 200, 200);
        doc.line(20, yPos, 190, yPos);
        yPos += 8;

        doc.setFontSize(8);
        doc.setTextColor(150, 150, 150);
        const disclaimer = 'MEDICAL DISCLAIMER: HealthScope utilizes advanced machine learning models to generate health risk assessments based on statistical probabilities. These insights are for informational purposes only and do not constitute a medical diagnosis. Please consult a certified healthcare professional for comprehensive clinical evaluation.';
        const splitDisclaimer = doc.splitTextToSize(disclaimer, 170);
        doc.text(splitDisclaimer, 20, yPos);

        // Save
        doc.save('HealthScope_Report.pdf');
    };

    return (
        <div className="min-h-screen bg-[#080c14] text-white selection:bg-amber-500/30 font-sans relative overflow-x-hidden">

            {/* Background */}
            <div className="fixed inset-0 z-0 pointer-events-none">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_var(--tw-gradient-stops))] from-blue-900/10 via-[#080c14] to-[#080c14]"></div>
            </div>

            {/* Nav */}
            <nav className="fixed top-0 w-full z-50 border-b border-white/5 bg-[#080c14]/80 backdrop-blur-xl">
                <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <Link href="/" className="flex items-center gap-3 group">
                            <Image src="/logo.png" alt="HealthScope Logo" width={28} height={28} className="group-hover:scale-110 transition-transform" />
                            <span className="text-xl font-bold tracking-tight text-white hover:text-blue-400 transition-colors">HealthScope</span>
                        </Link>
                    </div>
                    <div className="flex items-center gap-6">
                        <Link href="/" className="text-sm text-slate-400 hover:text-white transition-colors hover:underline cursor-pointer flex items-center gap-2">
                            <Home size={14} />
                            Back to Home
                        </Link>
                        <button onClick={() => router.push('/form')} className="text-sm text-slate-400 hover:text-white transition-colors flex items-center gap-2 hover:underline cursor-pointer">
                            <RefreshCw size={14} /> New Analysis
                        </button>
                    </div>
                </div>
            </nav>

            <main className="relative z-10 pt-32 pb-20 container mx-auto px-6 max-w-7xl transform scale-[0.98] origin-top">

                {/* Header Area */}
                <div className="text-left space-y-4 mb-16">
                    <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-300 text-xs font-bold uppercase tracking-widest">
                        <BrainCircuit size={14} /> Risk Analysis Complete
                    </div>
                    <h1 className="text-5xl md:text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-b from-white to-slate-400">
                        Your Health Analysis
                    </h1>
                </div>

                {/* Risk Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">

                    {/* 1. Diabetes Card */}
                    <div className={`group relative p-6 rounded-3xl backdrop-blur-sm border transition-all duration-300 ${premiumCardStyle}`}>
                        <div className="flex items-center justify-between mb-8">
                            <div className={`p-4 rounded-xl border ${premiumIconStyle}`}>
                                <Activity size={24} />
                            </div>
                            <span className={`text-xs font-bold uppercase tracking-wider px-3 py-1.5 rounded-lg ${premiumBadgeStyle}`}>
                                {getRiskLabel(results.diabetes_risk)}
                            </span>
                        </div>
                        <div className="text-blue-500/60 text-xs uppercase tracking-widest font-bold mb-2">Diabetes Probability</div>
                        <div className="text-5xl font-bold text-white mb-4">{(results.diabetes_risk * 100).toFixed(1)}%</div>
                        <p className="text-slate-400 text-sm leading-relaxed border-t border-blue-500/10 pt-4">
                            Diabetes risk assessment based on health indicators and lifestyle patterns.
                        </p>
                    </div>

                    {/* 2. Heart Card */}
                    <div className={`group relative p-6 rounded-3xl backdrop-blur-sm border transition-all duration-300 ${premiumCardStyle}`}>
                        <div className="flex items-center justify-between mb-8">
                            <div className={`p-4 rounded-xl border ${premiumIconStyle}`}>
                                <Heart size={24} />
                            </div>
                            <span className={`text-xs font-bold uppercase tracking-wider px-3 py-1.5 rounded-lg ${premiumBadgeStyle}`}>
                                {getRiskLabel(results.heart_risk)}
                            </span>
                        </div>
                        <div className="text-blue-500/60 text-xs uppercase tracking-widest font-bold mb-2">Cardiac Probability</div>
                        <div className="text-5xl font-bold text-white mb-4">{(results.heart_risk * 100).toFixed(1)}%</div>
                        <p className="text-slate-400 text-sm leading-relaxed border-t border-blue-500/10 pt-4">
                            Analysis of cardiovascular stress markers and lifestyle factors.
                        </p>
                    </div>

                    {/* 3. Obesity Card */}
                    <div className={`group relative p-6 rounded-3xl backdrop-blur-sm border transition-all duration-300 ${premiumCardStyle}`}>
                        <div className="flex items-center justify-between mb-8">
                            <div className={`p-4 rounded-xl border ${premiumIconStyle}`}>
                                <Scale size={24} />
                            </div>
                            <span className={`text-xs font-bold uppercase tracking-wider px-3 py-1.5 rounded-lg ${premiumBadgeStyle}`}>
                                {results.bmi.toFixed(1)} BMI
                            </span>
                        </div>
                        <div className="text-blue-500/60 text-xs uppercase tracking-widest font-bold mb-2">Obesity Level</div>
                        <div className="text-[42px] font-bold text-white mb-4 tracking-tighter leading-none" title={results.obesity_level}>
                            {results.obesity_level.replace('Insufficient_Weight', 'Underweight')
                                .replace('Normal_Weight', 'Healthy Weight')
                                .replace('Overweight_Level_I', 'Overweight I')
                                .replace('Overweight_Level_II', 'Overweight II')
                                .replace('Obesity_Type_I', 'Moderate Obesity')
                                .replace('Obesity_Type_II', 'Severe Obesity')
                                .replace('Obesity_Type_III', 'Very Severe Obesity')
                                .replace(/ll/g, '')
                                .replace(/II/g, '')
                                .replace(/_/g, ' ')}
                        </div>
                        <p className="text-slate-400 text-sm leading-relaxed border-t border-blue-500/10 pt-4">
                            Categorization derived from multi-factor analysis of BMI, diet, and physical activity.
                        </p>
                    </div>
                </div>

                {/* Risk Overview Chart Section */}
                <div className="mb-16">
                    <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8">
                        <h3 className="text-2xl font-bold text-white mb-8 flex items-center gap-3">
                            <div className="w-1 h-8 bg-gradient-to-b from-blue-400 to-blue-600 rounded-full"></div>
                            Risk Overview
                        </h3>

                        <div className="space-y-6">
                            {/* Diabetes Risk Bar */}
                            <div className="space-y-2">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <Activity size={20} className="text-blue-400" />
                                        <span className="text-sm font-bold text-slate-300">Diabetes Risk</span>
                                    </div>
                                    <span className="text-lg font-bold text-blue-400">{(results.diabetes_risk * 100).toFixed(1)}%</span>
                                </div>
                                <div className="h-3 bg-slate-800/50 rounded-full overflow-hidden border border-white/5">
                                    <div
                                        className="h-full bg-gradient-to-r from-blue-500 to-blue-400 rounded-full transition-all duration-1000 ease-out"
                                        style={{ width: `${results.diabetes_risk * 100}%` }}
                                    />
                                </div>
                                <div className="text-xs text-slate-500 text-right">{getRiskLabel(results.diabetes_risk)}</div>
                            </div>

                            {/* Heart Risk Bar */}
                            <div className="space-y-2">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <Heart size={20} className="text-blue-400" />
                                        <span className="text-sm font-bold text-slate-300">Cardiac Risk</span>
                                    </div>
                                    <span className="text-lg font-bold text-blue-400">{(results.heart_risk * 100).toFixed(1)}%</span>
                                </div>
                                <div className="h-3 bg-slate-800/50 rounded-full overflow-hidden border border-white/5">
                                    <div
                                        className="h-full bg-gradient-to-r from-blue-500 to-blue-400 rounded-full transition-all duration-1000 ease-out"
                                        style={{ width: `${results.heart_risk * 100}%` }}
                                    />
                                </div>
                                <div className="text-xs text-slate-500 text-right">{getRiskLabel(results.heart_risk)}</div>
                            </div>

                            {/* BMI Indicator */}
                            <div className="space-y-2">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <Scale size={20} className="text-blue-400" />
                                        <span className="text-sm font-bold text-slate-300">Body Mass Index</span>
                                    </div>
                                    <span className="text-lg font-bold text-blue-400">{results.bmi.toFixed(1)}</span>
                                </div>
                                <div className="h-3 bg-slate-800/50 rounded-full overflow-hidden border border-white/5 relative">
                                    {/* Normal range indicator (18.5-24.9) */}
                                    <div className="absolute left-[46%] w-[16%] h-full bg-green-500/20"></div>
                                    <div
                                        className="h-full bg-gradient-to-r from-blue-500 to-blue-400 rounded-full transition-all duration-1000 ease-out"
                                        style={{ width: `${Math.min((results.bmi / 40) * 100, 100)}%` }}
                                    />
                                </div>
                                <div className="flex justify-between text-xs text-slate-500">
                                    <span>Underweight</span>
                                    <span className="text-green-400">Normal (18.5-24.9)</span>
                                    <span>Obese</span>
                                </div>
                            </div>
                        </div>

                        {/* Overall Health Score Summary */}
                        <div className="mt-8 pt-6 border-t border-white/10 flex items-center justify-between">
                            <span className="text-sm font-bold text-slate-400 uppercase tracking-wider">Overall Health Score</span>
                            <div className="flex items-baseline gap-2">
                                <span className="text-4xl font-bold text-blue-400">{overallScore}</span>
                                <span className="text-lg text-blue-500/40 font-medium">/100</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Detailed Breakdown */}
                <div className="mb-16 mt-24">
                    <h3 className="text-3xl font-bold text-white mb-12 flex items-center justify-center gap-3">
                        <FileText size={32} className="text-blue-400" /> Detailed Recommendations
                    </h3>

                    <div className="space-y-4 max-w-4xl mx-auto">
                        {/* Expandable Item 1 */}
                        <div className="bg-[#0e1629]/50 border border-white/5 rounded-2xl overflow-hidden hover:border-blue-500/30 transition-colors">
                            <button onClick={() => setOpenSection(openSection === 'diabetes' ? null : 'diabetes')}
                                className="w-full flex items-center justify-between p-6 text-left cursor-pointer">
                                <div className="flex items-center gap-4">
                                    <div className="w-1.5 h-12 rounded-full bg-gradient-to-b from-blue-400 to-blue-600"></div>
                                    <div>
                                        <div className="font-bold text-white text-lg">Diabetes Management</div>
                                        <div className="text-xs text-blue-500/70 uppercase tracking-widest">Clinical Protocol</div>
                                    </div>
                                </div>
                                {openSection === 'diabetes' ? <ChevronUp className="text-slate-500" /> : <ChevronDown className="text-slate-500" />}
                            </button>
                            {openSection === 'diabetes' && (
                                <div className="px-8 pb-8 pt-2 text-slate-400 text-sm leading-relaxed border-t border-white/5">
                                    <div className="pt-4 space-y-3">
                                        <p className="flex gap-3"><CheckCircle size={18} className="text-blue-500 mt-0.5" /> Monitor blood glucose levels regularly (Fasting & Post-prandial).</p>
                                        <p className="flex gap-3"><CheckCircle size={18} className="text-blue-500 mt-0.5" /> Maintain a low-glycemic index diet rich in fiber.</p>
                                        <p className="flex gap-3"><CheckCircle size={18} className="text-blue-500 mt-0.5" /> Aim for 150 minutes of moderate-intensity aerobic activity per week.</p>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Expandable Item 2 */}
                        <div className="bg-[#0e1629]/50 border border-white/5 rounded-2xl overflow-hidden hover:border-blue-500/30 transition-colors">
                            <button onClick={() => setOpenSection(openSection === 'heart' ? null : 'heart')}
                                className="w-full flex items-center justify-between p-6 text-left cursor-pointer">
                                <div className="flex items-center gap-4">
                                    <div className="w-1.5 h-12 rounded-full bg-gradient-to-b from-blue-400 to-blue-600"></div>
                                    <div>
                                        <div className="font-bold text-white text-lg">Cardiovascular Health</div>
                                        <div className="text-xs text-blue-500/70 uppercase tracking-widest">Preventative Care</div>
                                    </div>
                                </div>
                                {openSection === 'heart' ? <ChevronUp className="text-slate-500" /> : <ChevronDown className="text-slate-500" />}
                            </button>
                            {openSection === 'heart' && (
                                <div className="px-8 pb-8 pt-2 text-slate-400 text-sm leading-relaxed border-t border-white/5">
                                    <div className="pt-4 space-y-3">
                                        <p className="flex gap-3"><CheckCircle size={18} className="text-blue-500 mt-0.5" /> Regular monitoring of Blood Pressure and Lipid Profile.</p>
                                        <p className="flex gap-3"><CheckCircle size={18} className="text-blue-500 mt-0.5" /> Adopt a DASH or Mediterranean diet (low sodium, healthy fats).</p>
                                        <p className="flex gap-3"><CheckCircle size={18} className="text-blue-500 mt-0.5" /> Practice stress-reduction techniques (meditation, adequate sleep).</p>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Expandable Item 3 */}
                        <div className="bg-[#0e1629]/50 border border-white/5 rounded-2xl overflow-hidden hover:border-blue-500/30 transition-colors">
                            <button onClick={() => setOpenSection(openSection === 'obesity' ? null : 'obesity')}
                                className="w-full flex items-center justify-between p-6 text-left cursor-pointer">
                                <div className="flex items-center gap-4">
                                    <div className="w-1.5 h-12 rounded-full bg-gradient-to-b from-blue-400 to-blue-600"></div>
                                    <div>
                                        <div className="font-bold text-white text-lg">Weight Management</div>
                                        <div className="text-xs text-blue-500/70 uppercase tracking-widest">Metabolic Health</div>
                                    </div>
                                </div>
                                {openSection === 'obesity' ? <ChevronUp className="text-slate-500" /> : <ChevronDown className="text-slate-500" />}
                            </button>
                            {openSection === 'obesity' && (
                                <div className="px-8 pb-8 pt-2 text-slate-400 text-sm leading-relaxed border-t border-white/5">
                                    <div className="pt-4 space-y-3">
                                        <p className="flex gap-3"><CheckCircle size={18} className="text-blue-500 mt-0.5" /> <strong>BMI {results.bmi.toFixed(1)}:</strong> Classified as {results.obesity_level}.</p>
                                        <p className="flex gap-3"><CheckCircle size={18} className="text-blue-500 mt-0.5" /> Balance caloric intake with expenditure (caloric deficit for weight loss).</p>
                                        <p className="flex gap-3"><CheckCircle size={18} className="text-blue-500 mt-0.5" /> Incorporate strength training to improve metabolic rate.</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Bottom Action: Export */}
                <div className="flex justify-center mb-16">
                    <div className="w-full max-w-2xl bg-blue-500/5 border border-blue-500/10 rounded-3xl p-8 text-center backdrop-blur-md">
                        <h4 className="text-xl font-bold text-white mb-2">Ready to take action?</h4>
                        <p className="text-slate-400 text-sm mb-6 max-w-md mx-auto">
                            Download your comprehensive health report to share with your healthcare provider.
                        </p>
                        <button onClick={downloadReport} className="py-4 px-12 bg-blue-500 hover:bg-blue-400 text-white font-bold rounded-xl shadow-[0_0_30px_-10px_rgba(59,130,246,0.5)] transition-all flex items-center justify-center gap-3 mx-auto cursor-pointer">
                            <Download size={20} /> Download Full Report
                        </button>
                    </div>
                </div>

                {/* Disclaimer Strip */}
                <div className="max-w-5xl mx-auto p-4 rounded-xl bg-white/5 border border-white/5 text-center">
                    <p className="text-xs text-amber-200/80 leading-relaxed text-center">
                        <span className="font-bold text-amber-400">DISCLAIMER:</span> HealthScope utilizes advanced machine learning models to generate health risk assessments based on statistical probabilities. These insights are for informational purposes only and do not constitute a medical diagnosis; kindly consult a certified healthcare professional for a comprehensive clinical evaluation.
                    </p>
                </div>

            </main>
        </div>
    );
}
