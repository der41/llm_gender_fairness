// src/app/page.tsx
import Link from 'next/link';
import { getProfessions } from '@/lib/data';
import Image from 'next/image';

export default function Home() {
  const professions = getProfessions();

  return (
    <main className="min-h-screen bg-slate-50 p-10">
      <header className="mb-12 text-center">
        <h1 className="text-4xl font-extrabold text-slate-800 tracking-tight">
          How AI Imagines <span className="text-blue-600">Professions</span>
        </h1>
        <p className="text-slate-500 mt-2">Select a profession to view the gender bias analysis</p>
      </header>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6 max-w-7xl mx-auto">
        {professions.map((prof) => (
          <Link key={prof} href={`/${prof}`} className="group">
            <div className="bg-white rounded-xl overflow-hidden shadow-sm border border-slate-200 hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
              {/* Image Preview */}
              <div className="h-48 relative bg-slate-100 overflow-hidden">
                 {/* Using the actual wordclouds as previews */}
                 <Image 
                   src={`/results/wordcloud/${prof}_wordcloud.png`} 
                   alt={prof}
                   fill
                   className="object-cover opacity-80 group-hover:opacity-100 group-hover:scale-105 transition-all duration-500"
                 />
              </div>
              
              <div className="p-4 flex justify-between items-center bg-white relative z-10">
                <h2 className="text-xl font-bold text-slate-800 capitalize">{prof}</h2>
                <span className="text-slate-300 group-hover:text-blue-500 transition-colors">→</span>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </main>
  );
}