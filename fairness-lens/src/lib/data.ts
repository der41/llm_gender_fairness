import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';

const DATA_DIR = path.join(process.cwd(), 'public/results/data');

export function getProfessions() {
  if (!fs.existsSync(DATA_DIR)) return [];
  
  const files = fs.readdirSync(DATA_DIR);
  const professions = new Set<string>();
  
  // excluded filenames or prefixes that look like professions but aren't
  const IGNORED = ['pca', 'all']; 

  files.forEach(file => {
    // Matches "name_" pattern (e.g. "doctor_judge_scores.csv")
    const match = file.match(/^([a-z]+)_/);
    if (match) {
      const profession = match[1];
      // Only add if it's not in our ignored list
      if (!IGNORED.includes(profession)) {
        professions.add(profession);
      }
    }
  });

  return Array.from(professions).sort();
}

export async function getProfessionData(profession: string) {
  // 1. Load Judge Summary (JSON)
  let judgeSummary = null;
  try {
    const jsonPath = path.join(DATA_DIR, `${profession}_final_bias_summary.json`);
    if (fs.existsSync(jsonPath)) {
      const fileContent = fs.readFileSync(jsonPath, 'utf8');
      judgeSummary = JSON.parse(fileContent);
    }
  } catch (e) { console.log("No judge summary found"); }

  // 2. Load Gender Frequency (CSV)
  let genderStats = [
    { name: 'Male', value: 0, original: 0, fill: '#636EFA' },
    { name: 'Female', value: 0, original: 0, fill: '#EF553B' },
    { name: 'Neutral', value: 0, original: 0, fill: '#94a3b8' },
  ];

  try {
    const csvPath = path.join(DATA_DIR, `${profession}_gender_freq.csv`);
    if (fs.existsSync(csvPath)) {
      const csvFile = fs.readFileSync(csvPath, 'utf8');
      const parsed = Papa.parse(csvFile, { header: true, skipEmptyLines: true });
      
      parsed.data.forEach((row: any) => {
        const label = row['gender_label']?.toLowerCase();
        const count = parseFloat(row['count']);

        if (!isNaN(count)) {
          if (label === 'male') {
            genderStats[0].value += (count * 10); 
            genderStats[0].original += count;     
          } else if (label === 'female') {
            genderStats[1].value += (count * 10); 
            genderStats[1].original += count;     
          } else if (label && (label.includes('non-gender') || label.includes('neutral'))) {
            genderStats[2].value += count;        
            genderStats[2].original += count;
          }
        }
      });
    }
  } catch (e) { console.log("Error parsing gender stats", e); }

  // 3. Load Sample Text (CSV)
  let sampleText = "No sample available.";
  try {
    const samplePath = path.join(DATA_DIR, `${profession}_samples_paragraphs.csv`);
    if (fs.existsSync(samplePath)) {
      const csvFile = fs.readFileSync(samplePath, 'utf8');
      const parsed = Papa.parse(csvFile, { header: true, skipEmptyLines: true });
      if (parsed.data.length > 0) {
          // @ts-ignore
          sampleText = parsed.data[0].paragraph || parsed.data[0].text; 
      }
    }
  } catch (e) {}

   // 4. Load Top Adjectives (CSV)
   let topAdjectives: { adjective: string; count: number }[] = [];
   try {
     const adjPath = path.join(DATA_DIR, `${profession}_adjectives_freq.csv`);
     if (fs.existsSync(adjPath)) {
       const csvFile = fs.readFileSync(adjPath, 'utf8');
       const parsed = Papa.parse(csvFile, { header: true, skipEmptyLines: true, dynamicTyping: true });
       
       if(parsed.data && Array.isArray(parsed.data)) {
          topAdjectives = (parsed.data as any[])
             .filter(row => row.adjective && typeof row.count === 'number')
             .sort((a, b) => b.count - a.count)
             .slice(0, 4)
             .map(row => ({ adjective: row.adjective, count: row.count }));
       }
     }
   } catch (e) { console.error("Error loading adjectives:", e); }

  // 5. Load PCA Circles (JSON)
  let embeddingCircles = null;
  try {
    const pcaPath = path.join(DATA_DIR, 'pca_circles.json');
    if (fs.existsSync(pcaPath)) {
       const fileContent = fs.readFileSync(pcaPath, 'utf8');
       const allCircles = JSON.parse(fileContent);
       embeddingCircles = allCircles[profession] || null;
    }
  } catch (e) { console.log("No PCA data found"); }

  return { judgeSummary, genderStats, sampleText, topAdjectives, embeddingCircles };
}