/** Read-only release checks. Installing/rolling back needs a managed installer. */
const RELEASES = 'https://api.github.com/repos/Niko1221/Strata/releases?per_page=30';
const RELEASE_PREFIX = 'https://github.com/Niko1221/Strata/releases/';

function versionParts(value) {
  const match = /^v?(\d+)\.(\d+)\.(\d+)(?:\+[\w.-]+)?$/.exec(value ?? '');
  return match ? match.slice(1).map(Number) : null;
}
function compare(a, b) {
  for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] > b[i] ? 1 : -1;
  return 0;
}

export async function checkUpdates({ appVersion, engineVersion }, fetcher = fetch) {
  let strata;
  try {
    const response = await fetcher(RELEASES, { headers: { accept: 'application/vnd.github+json' }, signal: AbortSignal.timeout(10000) });
    if (!response.ok) throw new Error(response.status === 403 || response.status === 429 ? 'The release service is busy. Try again later.' : 'Could not reach the Strata release feed. Try again.');
    const releases = await response.json();
    if (!Array.isArray(releases)) throw new Error('The release service returned an invalid response.');
    const candidates = releases.filter(r => !r.draft && !r.prerelease && versionParts(r.tag_name));
    candidates.sort((a, b) => compare(versionParts(b.tag_name), versionParts(a.tag_name)));
    const latest = candidates[0];
    const installed = versionParts(appVersion);
    if (!latest) strata = { status: 'no-releases', message: 'No stable Strata release is published in this feed yet.' };
    else if (!installed) strata = { status: 'unavailable', message: 'This installation has no comparable version number.' };
    else {
      const newer = compare(versionParts(latest.tag_name), installed) > 0;
      // Release content is untrusted. Only link back to the known repository.
      const releaseUrl = typeof latest.html_url === 'string' && latest.html_url.startsWith(RELEASE_PREFIX) ? latest.html_url : undefined;
      strata = { status: newer ? 'available' : 'current', latestVersion: latest.tag_name, releaseUrl, message: newer ? `Strata ${latest.tag_name} is available.` : 'Strata is up to date.' };
    }
  } catch (error) {
    strata = { status: 'error', message: error.name === 'TimeoutError' ? 'The release check timed out. Try again.' : error.message ?? 'Could not check for updates.' };
  }
  return {
    checkedAt: new Date().toISOString(),
    strata: { installedVersion: appVersion, ...strata, canInstall: false },
    // The existing binary downloader has no tested-release compatibility feed
    // or trustworthy installed build ID. Upstream's latest is not an update verdict.
    engine: { installedVersion: engineVersion, status: 'unavailable', canInstall: false, message: 'Engine update checks are not available in this build.' },
  };
}
