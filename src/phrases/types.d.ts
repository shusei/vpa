export type PhraseDifficulty = "E" | "M" | "H";
export interface PhraseItem {
  id: string;
  cat: string;
  text: string;
  alts?: string[];
  tags?: string[];
  difficulty?: PhraseDifficulty;
  notes?: string;
}
export interface PhraseCategory {
  id: string;
  name: string;
}
export interface PhrasePack {
  id: string;
  lang: string;
  version: string;
  title: string;
  license: string;
  categories: PhraseCategory[];
  items: PhraseItem[];
}
