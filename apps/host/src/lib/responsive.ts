import { useEffect, useState } from 'react';

export function useMedia(query: string) {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const media = window.matchMedia(query);
    const update = () => setMatches(media.matches);
    update();
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, [query]);
  return matches;
}

export function scrollBehavior(): ScrollBehavior {
  const override = document.documentElement.dataset.motion;
  return override === 'reduced' || (override !== 'full' && window.matchMedia('(prefers-reduced-motion: reduce)').matches) ? 'instant' : 'smooth';
}
