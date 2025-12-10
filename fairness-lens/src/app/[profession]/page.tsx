import { getProfessionData, getProfessions } from '@/lib/data';
import Link from 'next/link';
import Image from 'next/image';
import { ChartComponent } from './chart';
import { ArrowLeft, Gavel, Quote } from 'lucide-react';

// 1. Correctly type params as a Promise for Next.js 15+
type Params = Promise<{ profession: string }>;

export async function generateStaticParams() {
  const professions = getProfessions();
  return professions.map((profession) => ({ profession }));
}

// 2. Ensure the component is the default export
export default async function ProfessionPage(props: { params: Params }) {
  // 3. Await the params before using them
  const params = await props.params;
  const { profession } = params;

  const data = await getProfessionData(profession);

  return (
    <div className="min-h-screen bg-slate-50 p-8 pb-20">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <Link href="/" className="inline-flex items-center text-slate-500 hover:text-blue-600 mb-6 transition-colors font-medium">
          <ArrowLeft className="w-4 h-4 mr-2" /> Back to Menu
        </Link>

        <h1 className="text-5xl font-black text-slate-900 capitalize mb-10 tracking-tight">{profession}</h1>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* LEFT COL: Visuals (Word Cloud) */}
          <div className="lg:col-span-7 space-y-6">
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
              <h3 className="text-lg font-semibold text-slate-400 uppercase tracking-wider mb-4">AI Imagination</h3>
              <div className="relative aspect-square w-full rounded-xl overflow-hidden bg-slate-100 border border-slate-100">
                <Image 
                  src={`/results/wordcloud/${profession}_wordcloud.png`}
                  alt="Word Cloud"
                  fill
                  className="object-contain p-4 hover:scale-105 transition-transform duration-700"
                />
              </div>
            </div>

             {/* Sample Text */}
             <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Quote className="w-5 h-5 text-blue-500" />
                <h3 className="text-lg font-semibold text-slate-800">Sample Generation</h3>
              </div>
              <p className="text-slate-600 italic leading-relaxed text-lg font-serif">
                "{data.sampleText}"
              </p>
            </div>
          </div>

          {/* RIGHT COL: Data & Judge */}
          <div className="lg:col-span-5 space-y-6">
            
            {/* Gender Frequency Chart */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
              <h3 className="text-lg font-semibold text-slate-800 mb-2">Gender Frequency</h3>
              <p className="text-sm text-slate-400 mb-6">Distribution of pronouns found in 1,000 descriptions</p>
              <div className="h-64 w-full">
                <ChartComponent data={data.genderStats} />
              </div>
            </div>

            {/* AI Judge Verdict */}
            <div className="bg-gradient-to-br from-indigo-600 to-blue-700 rounded-2xl shadow-lg text-white p-8 relative overflow-hidden">
               <Gavel className="absolute top-4 right-4 text-white/10 w-24 h-24 -rotate-12" />
               
               <h3 className="text-indigo-100 font-semibold uppercase tracking-wider text-sm mb-1">Gemini 2.5 Judge</h3>
               <h2 className="text-3xl font-bold mb-4">
                 {data.judgeSummary ? "Bias Detected" : "No Verdict"}
               </h2>
               
               {data.judgeSummary && (
                 <>
                   <div className="bg-white/10 backdrop-blur-md rounded-lg p-4 mb-4 border border-white/20">
                     <p className="text-lg font-medium leading-relaxed">
                       {data.judgeSummary.explanation || data.judgeSummary.final_decision}
                     </p>
                   </div>
                 </>
               )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}