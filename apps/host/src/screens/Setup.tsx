import { tr } from '../lib/i18n';
import { GuidedSetup } from '../components/GuidedSetup';
import { useNative } from '../lib/native';
import { useStore } from '../lib/store';
/** Every setup entry uses the same guided onboarding. */
export function SetupScreen() {
    const native = useNative(), setup = useStore(s => s.setup);
    if (!native || !setup)
        return <div className="setup guided-onboarding"><p className="onboard-loading" role="status">{tr("Getting your setup ready\u2026")}</p></div>;
    return <GuidedSetup />;
}
