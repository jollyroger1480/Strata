import english from '../locales/en.json';
import zh from '../locales/zh.json';
import ja from '../locales/ja.json';
import de from '../locales/de.json';
import es from '../locales/es.json';
import ru from '../locales/ru.json';
import pt from '../locales/pt.json';
import ko from '../locales/ko.json';

export const LANGUAGES=[{value:'en',label:'English'},{value:'zh',label:'中文（简体）'},{value:'ja',label:'日本語'},{value:'de',label:'Deutsch'},{value:'es',label:'Español'},{value:'ru',label:'Русский'},{value:'pt',label:'Português'},{value:'ko',label:'한국어'}] as const;
export type Language=typeof LANGUAGES[number]['value'];
const catalogs:Record<Language,Record<string,string>>={en:english,zh,ja,de,es,ru,pt,ko};
const entities:Record<string,string>={amp:'&',lt:'<',gt:'>',quot:'"',apos:"'",nbsp:'\u00a0'};
const decode=(text:string)=>text.replace(/&(#x[\da-f]+|#\d+|amp|lt|gt|quot|apos|nbsp);/gi,(whole,key:string)=>key[0]==='#'?String.fromCodePoint(key[1].toLowerCase()==='x'?parseInt(key.slice(2),16):parseInt(key.slice(1),10)):entities[key]??whole);
const ids=new Map(Object.entries(english).map(([id,value])=>[decode(value),id]));
let language:Language='en';
export const validLanguage=(value:unknown):value is Language=>LANGUAGES.some(item=>item.value===value);
export const currentLanguage=()=>language;
export const localeTag=()=>language==='zh'?'zh-CN':language;
export function setTranslationLanguage(value:Language){language=value;}
const escape=(value:string)=>value.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
const patterns=Object.entries(english).filter(([,value])=>/\{\d+\}/.test(value)).map(([id,value])=>{
 const slots:number[]=[];let expression='',at=0;
 for(const match of value.matchAll(/\{(\d+)\}/g)){expression+=escape(value.slice(at,match.index))+'([\\s\\S]*?)';slots.push(Number(match[1]));at=match.index!+match[0].length;}
 expression+=escape(value.slice(at));
 return{id,slots,pattern:new RegExp('^'+expression+'$'),specificity:value.replace(/\{\d+\}/g,'').length};
}).sort((a,b)=>b.specificity-a.specificity);
const localizedTemplateSlots:Record<string,number[]>={s2473:[1,2],s0825:[0,1],s2455:[0],s2456:[0]};

/** Translate interface copy only. Non-string React children pass through unchanged. */
export function tr<T>(value:T,values?:readonly unknown[]):T{
 if(typeof value!=='string')return value;
 const leading=value.match(/^\s*/)?.[0]??'',trailing=value.match(/\s*$/)?.[0]??'',key=decode(value.trim());
 const dictionary=catalogs[language]??english;
 let id=ids.get(key),slots=values;
 if(!id&&language!=='en')for(const candidate of patterns){const match=candidate.pattern.exec(key);if(match){id=candidate.id;const found:unknown[]=[];candidate.slots.forEach((slot,index)=>{const captured=match[index+1];found[slot]=localizedTemplateSlots[candidate.id]?.includes(slot)?tr(captured):captured;});slots=found;break;}}
 let text=decode(id?(dictionary[id]??english[id as keyof typeof english]):key);
 if(slots)text=text.replace(/\{(\d+)\}/g,(_,index)=>String(slots![Number(index)]??''));
 if(!id&&!values)return decode(value) as T;
 return (leading+text+trailing) as T;
}
