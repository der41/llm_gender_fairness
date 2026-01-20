import { getProfessions } from '@/lib/data';
import ClientHome from './ClientHome';

type SearchParams = Promise<{ [key: string]: string | string[] | undefined }>;

export default async function Home(props: { searchParams: SearchParams }) {
  const professions = getProfessions();
  const searchParams = await props.searchParams;
  
  // Check if "menu=true" exists in the URL
  const startAtMenu = searchParams.menu === 'true';

  return <ClientHome professions={professions} startAtMenu={startAtMenu} />;
}