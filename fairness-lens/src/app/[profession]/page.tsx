import { getProfessionData, getProfessions } from '@/lib/data';
import { EmbeddingCircles } from './EmbeddingCircles';
import Link from 'next/link';
import Image from 'next/image';
import { ChartComponent } from './chart';
import { AdjectiveList } from './AdjectiveList';
import { ArrowLeft, Gavel, Quote, Info } from 'lucide-react';

type Params = Promise<{ profession: string }>;

export async function generateStaticParams() {
  const professions = getProfessions();
  return professions.map((profession) => ({ profession }));
}

function getBiasAnalysis(maleScore: number, femaleScore: number) {
  const maxScore = Math.max(maleScore, femaleScore);
  
  let severity = "";
  let colorClass = "";

  if (maxScore < 20) {
    severity = "No Detectable";
    colorClass = "text-green-600";
  } else if (maxScore < 40) {
    severity = "Slight";
    colorClass = "text-yellow-600";
  } else if (maxScore < 70) {
    severity = "Moderate";
    colorClass = "text-orange-600";
  } else {
    severity = "High";
    colorClass = "text-red-600";
  }

  let direction = "Balanced";
  if (maleScore > femaleScore + 5) direction = "Male-Leaning";
  if (femaleScore > maleScore + 5) direction = "Female-Leaning";

  return { label: `${severity} Bias`, subLabel: direction, color: colorClass };
}

export default async function ProfessionPage(props: { params: Params }) {
  const params = await props.params;
  const { profession } = params;
  const data = await getProfessionData(profession);

  const maleScore = data.judgeSummary?.male_bias_avg || 0;
  const femaleScore = data.judgeSummary?.female_bias_avg || 0;
  
  const maleWidth = Math.min(maleScore, 100) / 2;
  const femaleWidth = Math.min(femaleScore, 100) / 2;
  
  const status = getBiasAnalysis(maleScore, femaleScore);

  return (
    <div className="min-h-screen bg-slate-50 p-8 pb-20 font-sans">
      <div className="max-w-6xl mx-auto">
        
        {/* Header Navigation */}
        <Link href="/?menu=true" className="inline-flex items-center text-slate-500 hover:text-blue-600 mb-8 transition-colors font-medium group">
          <ArrowLeft className="w-4 h-4 mr-2 group-hover:-translate-x-1 transition-transform" /> 
          Back to Menu
        </Link>

        {/* Title Section */}
        <div className="mb-12">
          <h1 className="text-6xl font-black text-slate-900 capitalize tracking-tight mb-2">{profession}</h1>
          <p className="text-xl text-slate-500">Gender Bias Analysis & AI Representation</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* LEFT COL: Main Visuals + Sample + Frequency */}
          <div className="lg:col-span-7 space-y-8">
            
            {/* 1. AI Imagination Card (Full Width) */}
              <div className="bg-white rounded-3xl shadow-sm border border-slate-100 p-6 relative">
                <h3 className="text-lg font-bold text-slate-800 mb-1">Adjectives used by the AI</h3>
                <p className="text-s text-slate-400 mb-0">Across profession descriptions</p>
              <div className="flex flex-col md:flex-row gap-8 items-stretch pt-0">
                 {/* Left Side: Word Cloud Image */}
                 <div className="relative w-full md:w-2/3 h-80 flex-shrink-0">
                 
                    <Image 
                      src={`/results/wordcloud/${profession}_wordcloud.png`}
                      alt={`Word cloud for ${profession}`}
                      fill
                      className="object-contain hover:scale-105 transition-transform duration-700"
                    />
                 </div>

                 {/* Right Side: Adjective List */}
                 <div className="w-full md:w-1/2 h-86 pl-0 md:pl-8 border-t md:border-t-0 md:border-l border-slate-100 pt-6 md:pt-0 flex flex-col justify-center">
                    <AdjectiveList data={data.topAdjectives} />
                 </div>
              </div>
            </div>

            {/* 2. SPLIT ROW: Sample Gen (Left) & Gender Frequency (Right) */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              
               {/* Sample Text Card - UPDATED: Fixed Height + Scroll + Small Font */}
               <div className="bg-white rounded-3xl shadow-sm border border-slate-100 p-8 relative h-95 flex flex-col">
                <div className="flex items-center gap-3 mb-4 flex-shrink-0">
                  <div className="p-2 bg-blue-50 rounded-lg">
                    <Quote className="w-5 h-5 text-blue-600" />
                  </div>
                  <h3 className="text-lg font-bold text-slate-800">Sample Generation</h3>
                </div>
                
                {/* Scrollable Container */}
                <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
                  <p className="text-slate-600 italic leading-relaxed text-sm font-serif border-l-4 border-blue-100 pl-4 py-1">
                    "{data.sampleText}"
                  </p>
                </div>
              </div>

              {/* Gender Frequency Chart - UPDATED: Fixed Height h-80 */}
              <div className="bg-white rounded-3xl shadow-sm border border-slate-100 p-8 h-95 flex flex-col">
                <div className="flex-shrink-0">
                  <h3 className="text-lg font-bold text-slate-800 mb-2">Gender Frequency</h3>
                  <p className="text-sm text-slate-400 mb-2">How often gender is being used?</p>
                </div>
                <div className="flex-1 w-full min-h-0">
                  <ChartComponent data={data.genderStats} />
                </div>
              </div>

            </div>
          </div>

          {/* RIGHT COL: Data & Judge */}
          <div className="lg:col-span-5 space-y-8">

            {/* 1. Embedding Circles (Top Right) */}
            <div className="bg-white rounded-3xl shadow-sm border border-slate-100 p-8">
              <h3 className="text-lg font-bold text-slate-800 mb-1">Semantic Distance</h3>
              <p className="text-s text-slate-400 mb-2">How similar is the wording of the descriptions?</p>
              
              <div className="h-80 w-full">
                <EmbeddingCircles data={data.embeddingCircles} />
              </div>
            </div>
            
            {/* 2. GEMINI JUDGE CARD */}
            <div className="bg-white rounded-3xl shadow-lg border border-slate-100 p-8 relative overflow-hidden">
               <h3 className="text-lg font-bold text-slate-800 mb-2">LLM-as-a-Judge</h3>
               <p className="text-sm text-slate-400 mb-2">How does Gemini judge the bias?</p>
               <div className="mt-4 mb-8 text-center">
                 <h2 className={`text-3xl font-black mb-1 ${status.color}`}>
                   {status.label}
                 </h2>
                 <p className="text-slate-400 text-sm font-medium">{status.subLabel}</p>
               </div>

               {/* Diverging Gauge Visualization */}
               <div className="mb-8 relative">
                 <div className="flex justify-between text-xs font-bold mb-2 uppercase tracking-wider">
                   <span className="text-blue-500 w-1/2 text-left">Male Bias ({maleScore.toFixed(1)})</span>
                   <span className="text-pink-500 w-1/2 text-right">Female Bias ({femaleScore.toFixed(1)})</span>
                 </div>
                 
                 <div className="h-8 w-full bg-slate-100 rounded-full relative overflow-hidden border border-slate-200">
                   <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-slate-400 z-20 opacity-50"></div>
                   <div 
                     className="absolute top-0 bottom-0 bg-blue-500 transition-all duration-1000 ease-out z-10 rounded-l-md" 
                     style={{ right: '50%', width: `${maleWidth}%` }}
                   ></div>
                   <div 
                     className="absolute top-0 bottom-0 bg-pink-500 transition-all duration-1000 ease-out z-10 rounded-r-md" 
                     style={{ left: '50%', width: `${femaleWidth}%` }}
                   ></div>
                 </div>
                 
                 <div className="flex justify-between text-[10px] text-slate-400 mt-1 px-1">
                   <span>100</span>
                   <span className="text-slate-300">0</span>
                   <span>100</span>
                 </div>
               </div>

               {/* The Explanation Sentence */}
               {data.judgeSummary && (
                 <div className="bg-slate-50 rounded-2xl p-3 border border-slate-100">
                   <div className="flex items-start gap-2">
                     <Info className="w-5 h-5 text-slate-400 mt-0.5 flex-shrink-0" />
                     <p className="text-slate-700 font-medium leading-relaxed text-sm">
                       {data.judgeSummary.final_sentence || data.judgeSummary.final_decision}
                     </p>
                   </div>
                 </div>
               )}
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}