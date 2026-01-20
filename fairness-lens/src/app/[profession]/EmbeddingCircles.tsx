'use client';

type CircleData = { x: number; y: number; r: number } | null;

interface EmbeddingData {
  male: CircleData;
  female: CircleData;
  neutral: CircleData;
}

export function EmbeddingCircles({ data }: { data: EmbeddingData | null }) {
  if (!data) return null;

  const colors = {
    male: "#2B7FFF",    // Blue
    female: "#F7369A",  // Pink
    neutral: "#D1D1D1"  // Gray
  };

  // 1. Calculate Bounding Box
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const groups = ['neutral', 'male', 'female'] as const;

  groups.forEach(g => {
    const c = data[g];
    if (c) {
      minX = Math.min(minX, c.x - c.r);
      maxX = Math.max(maxX, c.x + c.r);
      minY = Math.min(minY, c.y - c.r);
      maxY = Math.max(maxY, c.y + c.r);
    }
  });

  const paddingX = (maxX - minX) * 0.15;
  const paddingY = (maxY - minY) * 0.15;
  minX -= paddingX; maxX += paddingX;
  minY -= paddingY; maxY += paddingY;

  const width = maxX - minX;
  const height = maxY - minY;

  return (
    <div className="w-full h-full flex flex-col items-center justify-center">
      <div className="relative w-full aspect-[3/2]">
        <svg 
          viewBox={`${minX} ${minY} ${width} ${height}`} 
          className="w-full h-70"
          preserveAspectRatio="xMidYMid meet"
        >
          {/* Loop through Neutral -> Male -> Female */}
          {groups.map(g => {
            const circle = data[g];
            if (!circle) return null;
            
            return (
              <g key={g}>
                {/* 1. Main Circle */}
                <circle
                  cx={circle.x}
                  cy={circle.y}
                  r={circle.r}
                  fill={colors[g]}
                  opacity={1}
                  stroke={colors[g]}
                  strokeWidth={width * 0.005} 
                />
                
                {/* 2. Centroid (Drawn immediately after its circle) */}
                <circle
                  cx={circle.x}
                  cy={circle.y}
                  r={width * 0.05} 
                  fill={colors[g]} 
                  opacity={1}
                  stroke={colors[g]}
                  strokeWidth={width * 0.005}
                />
              </g>
            );
          })}
        </svg>
      </div>
      <div className="flex gap-4 mt-2 text-s font-medium text-slate-400">
        <div className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#D1D1D1]"></span> Neutral</div>
        <div className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#2B7FFF]"></span> Male</div>
        <div className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-[#F7369A]"></span> Female</div>
      </div>
    </div>
  );
}