export function AdjectiveList({ data }: { data: { adjective: string; count: number }[] }) {
    if (!data || data.length === 0) return null;
  
    // Find the highest count to calculate relative percentages
    const maxCount = Math.max(...data.map(d => d.count));
  
    return (
      <div className="flex flex-col justify-center h-full w-full gap-8 py-4">
        {data.map((item) => {
          const widthPercentage = (item.count / maxCount) * 100;
          
          return (
            <div key={item.adjective} className="w-full group">
              {/* Minimal Label: Just the word */}
              <div className="mb-2 text-lg font-medium text-slate-600 capitalize group-hover:text-blue-600 transition-colors">
                  {item.adjective}
              </div>
              
              {/* Minimal Bar: Thinner, no background track distraction */}
              <div className="h-1.5 w-full bg-slate-50 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-blue-400 rounded-full opacity-60 group-hover:opacity-100 transition-all duration-700 ease-out"
                  style={{ width: `${widthPercentage}%` }}
                ></div>
              </div>
            </div>
          );
        })}
      </div>
    );
  }