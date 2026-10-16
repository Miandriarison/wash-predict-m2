/* WASH Predict : front-end (JavaScript natif + Bootstrap 5 + Chart.js).
 *
 * Le navigateur ne calcule rien : il envoie les paramètres à l'API (POST /api/simulation)
 * et dessine ce qu'elle renvoie. Tout texte venant de l'API est inséré avec textContent
 * (jamais innerHTML) pour éviter l'injection de code.
 */
(() => {
  'use strict';

  const $ = (sel) => document.querySelector(sel);
  const CHAMPS_SCENARIO = [];          // [{cle, input, sortie}]
  const COULEURS = {
    realisation: '#e8604c', prevision: '#3b4a8a', scenario: '#f39c12',
    lineaire: '#95a5a6', objectif: '#00b39f', latrines: '#8e44ad',
  };
  const NOMS_CAS = ['Maintenir le taux actuel', 'Tendance (MLP)', 'Avec scénario', "Atteindre l'objectif"];
  const COULEURS_CAS = [COULEURS.lineaire, COULEURS.prevision, COULEURS.scenario, COULEURS.objectif];

  let meta = null;
  let graphique = null;
  let minuteur = null;
  let controleur = null;
  let performanceChargee = false;
  let derniere = null;                 // dernière simulation reçue (pour redessiner le graphique au changement de thème)

  // ------------------------------------------------------------------ thème clair / sombre
  const estSombre = () => document.documentElement.getAttribute('data-bs-theme') === 'dark';

  function appliquerThemeGraphique() {
    if (typeof Chart === 'undefined') return;
    Chart.defaults.color = estSombre() ? '#c9d3e3' : '#44506a';
    Chart.defaults.borderColor = estSombre() ? 'rgba(255,255,255,0.12)' : 'rgba(30,45,80,0.12)';
  }

  function basculerTheme() {
    const sombre = !estSombre();
    document.documentElement.setAttribute('data-bs-theme', sombre ? 'dark' : 'light');
    try { localStorage.setItem('theme', sombre ? 'dark' : 'light'); } catch (_) { /* stockage indisponible */ }
    $('#bascule-theme').textContent = sombre ? '☀️' : '🌙';
    if (graphique) { graphique.destroy(); graphique = null; }
    if (derniere) afficherGraphique(derniere);
  }

  // ------------------------------------------------------------------ formats
  const nombre = (v, nd = 0) =>
    v == null ? '—' : Number(v).toLocaleString('fr-FR', { minimumFractionDigits: nd, maximumFractionDigits: nd });
  const pct = (v, nd = 1) => (v == null ? '—' : `${nombre(v, nd)} %`);
  const pts = (v, nd = 1) => (v == null ? '—' : `${v > 0 ? '+' : ''}${nombre(v, nd)} pts`);

  function el(tag, props = {}, ...enfants) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === 'class') e.className = v;
      else if (k === 'text') e.textContent = v;
      else if (k === 'style') e.style.cssText = v;
      else e.setAttribute(k, v);
    }
    for (const c of enfants) if (c != null) e.append(c);
    return e;
  }

  // ------------------------------------------------------------------ API
  async function api(url, options) {
    const rep = await fetch(url, options);
    if (!rep.ok) {
      let detail = '';
      try {
        const j = await rep.json();
        detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
      } catch (_) { /* corps non JSON */ }
      throw new Error(detail || `Erreur ${rep.status}`);
    }
    return rep.json();
  }

  function afficherErreur(message) {
    const zone = $('#alertes');
    zone.replaceChildren(el('div', { class: 'alert alert-danger', role: 'alert', text: message }));
  }
  const effacerErreur = () => $('#alertes').replaceChildren();

  // ------------------------------------------------------------------ formulaire
  function lireFormulaire() {
    const type = $('#type').value;
    const niveaux = Object.fromEntries(CHAMPS_SCENARIO.map((c) => [c.cle, Number(c.input.value) / 100]));
    let valeur = null;
    if (type === 'region') valeur = $('#region').value;
    if (type === 'commune') valeur = Number($('#commune').value);
    return {
      type_territoire: type,
      valeur,
      nb_annees: Number($('#nb_annees').value),
      objectif_final: Number($('#objectif').value),
      sensibilisation: niveaux.sensibilisation ?? 0,
      education: niveaux.education ?? 0,
      campagnes_sante: niveaux.campagnes_sante ?? 0,
      croissance_pct: Number($('#croissance').value),
      personnes_par_latrine: Number($('#personnes_par_latrine').value),
      latrines_par_an: Number($('#latrines_par_an').value || 0),
    };
  }

  function formulaireValide() {
    let ok = true;
    for (const champ of $('#formulaire').querySelectorAll('input[type=number]')) {
      const valide = champ.checkValidity();
      champ.classList.toggle('is-invalid', !valide);
      ok = ok && valide;
    }
    return ok && ($('#type').value !== 'commune' || $('#commune').value !== '');
  }

  function majVisibiliteTerritoire() {
    const type = $('#type').value;
    $('#bloc-region').hidden = type === 'national';
    $('#bloc-commune').hidden = type !== 'commune';
  }

  async function chargerCommunes() {
    const region = $('#region').value;
    const liste = await api(`/api/communes?region=${encodeURIComponent(region)}`);
    const select = $('#commune');
    select.replaceChildren(...liste.map((c) => el('option', { value: String(c.id), text: c.label })));
  }

  function construireCurseursScenario() {
    const zone = $('#curseurs-scenario');
    for (const s of meta.scenarios) {
      const id = `scen-${s.cle}`;
      const sortie = el('output', { class: 'fw-semibold text-nowrap ms-2', for: id, text: '0 %' });
      const input = el('input', { type: 'range', class: 'form-range', id, min: '0', max: '100', step: '10', value: '0' });
      input.addEventListener('input', () => { sortie.textContent = `${input.value} %`; });
      zone.append(el('div', { class: 'mb-2' },
        el('label', { for: id, class: 'form-label d-flex justify-content-between mb-0' }, el('span', { text: s.libelle }), sortie),
        input));
      CHAMPS_SCENARIO.push({ cle: s.cle, input, sortie });
    }
  }

  // ------------------------------------------------------------------ affichage
  function afficherIndicateurs(d) {
    const i = d.indicateurs;
    $('#kpi-lib-realisation').textContent = `Réalisation ${d.annee_depart}`;
    $('#kpi-realisation').textContent = pct(i.realisation_depart);
    $('#kpi-lib-prevision').textContent = `Prévision ${d.annee_cible} (MLP)`;
    $('#kpi-prevision').textContent = pct(i.prevision_cible);
    $('#kpi-lib-scenario').textContent = `Avec scénario ${d.annee_cible}`;
    $('#kpi-scenario').textContent = pct(i.scenario_cible);
    const gain = $('#kpi-gain');
    gain.textContent = pts(i.gain_scenario_pts);
    gain.className = `small ${i.gain_scenario_pts > 0 ? 'text-success' : 'text-body-secondary'}`;
    $('#kpi-lib-ecart').textContent = `Écart à l'objectif ${d.annee_cible}`;
    const ecart = $('#kpi-ecart');
    ecart.textContent = pts(i.ecart_objectif_pts);
    ecart.className = `valeur ${i.ecart_objectif_pts >= 0 ? 'text-success' : 'text-danger'}`;
    $('#effet').textContent = `+${nombre(d.effet * 100, 0)} %`;
    $('#annee-cible').textContent = String(d.annee_cible);
  }

  function jeuxDeDonnees(d) {
    const s = d.series;
    const base = { tension: 0, spanGaps: false, pointRadius: 3, pointHoverRadius: 5 };
    const jeux = [];
    const ajouter = (cle, label, style) => { if (s[cle]) jeux.push({ label, data: s[cle], ...base, ...style }); };
    ajouter('realisation', 'Réalisation', { borderColor: COULEURS.realisation, backgroundColor: COULEURS.realisation, borderWidth: 3 });
    ajouter('retro', 'Modèle sur le passé (prédiction à 1 an)', {
      borderColor: COULEURS.prevision, backgroundColor: COULEURS.prevision, showLine: false, pointStyle: 'crossRot', pointRadius: 6, borderWidth: 2,
    });
    ajouter('prevision', 'Prévision (MLP)', { borderColor: COULEURS.prevision, backgroundColor: COULEURS.prevision, borderDash: [8, 5], borderWidth: 2 });
    ajouter('scenario', 'Prévision avec scénario', { borderColor: COULEURS.scenario, backgroundColor: COULEURS.scenario, borderDash: [8, 5], borderWidth: 2 });
    ajouter('lineaire', 'Prévision linéaire (méthode actuelle)', { borderColor: COULEURS.lineaire, backgroundColor: COULEURS.lineaire, borderDash: [2, 4], borderWidth: 2, pointRadius: 2 });
    ajouter('latrines', 'Avec latrines construites', { borderColor: COULEURS.latrines, backgroundColor: COULEURS.latrines, borderDash: [10, 3, 2, 3], borderWidth: 2 });
    ajouter('objectif', 'Objectif', { borderColor: COULEURS.objectif, backgroundColor: COULEURS.objectif, borderDash: [8, 5], borderWidth: 2 });
    return jeux;
  }

  function afficherGraphique(d) {
    $('#titre-graphique').textContent = `Taux d'accès à l'assainissement de base — ${d.territoire.libelle}`;
    $('#note-retro').hidden = !d.series.retro;
    if (typeof Chart === 'undefined') {
      afficherErreur("Chart.js n'a pas pu être chargé (connexion internet ?) : le graphique est indisponible.");
      return;
    }
    appliquerThemeGraphique();
    const jeux = jeuxDeDonnees(d);
    if (graphique) {
      graphique.data.labels = d.annees;
      graphique.data.datasets = jeux;
      graphique.update();
      return;
    }
    graphique = new Chart($('#graphique'), {
      type: 'line',
      data: { labels: d.annees, datasets: jeux },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 250 },
        interaction: { mode: 'index', intersect: false },
        scales: {
          y: { min: 0, max: 100, title: { display: true, text: 'Taux en (%)' } },
          x: { title: { display: true, text: 'Année' } },
        },
        plugins: {
          legend: { position: 'bottom' },
          tooltip: { callbacks: { label: (c) => `${c.dataset.label} : ${nombre(c.parsed.y, 2)} %` } },
        },
      },
    });
  }

  function afficherBesoin(d) {
    const b = d.besoin;
    $('#besoin-intro').textContent =
      `D'ici ${b.annee_cible}, la population passe à environ ${nombre(b.population_cible)} habitants ` +
      `(+${nombre(Number($('#croissance').value), 1)} %/an). Par rapport à ${d.annee_depart}, il faut équiper en plus :`;

    const zone = $('#besoin-cas');
    zone.replaceChildren(...b.cas.map((c, k) =>
      el('div', { class: 'col-6 col-xl-3' },
        el('div', { class: 'card h-100 carte-cas', style: `--couleur:${COULEURS_CAS[k]}` },
          el('div', { class: 'card-body' },
            el('div', { class: 'small text-body-secondary', text: c.nom }),
            el('div', { class: 'valeur', text: `${nombre(c.latrines)} latrines` }),
            el('div', { class: 'small text-body-secondary', text: `${nombre(c.personnes)} personnes` }))))));

    const lat = $('#besoin-latrines');
    const sl = b.simulation_latrines;
    lat.hidden = !sl;
    if (sl) {
      lat.textContent =
        `Avec ${nombre(sl.latrines_par_an)} latrines/an : taux ${b.annee_cible} = ${pct(sl.taux_cible)} ` +
        `(${pts(sl.variation_pts)} par rapport à ${d.annee_depart}).`;
    }

    // tableau détaillé
    const table = $('#table-besoin');
    const entete = el('tr', {}, el('th', { class: 'text-start', text: 'Année' }), el('th', { text: 'Population' }));
    for (const nom of NOMS_CAS) {
      entete.append(el('th', { text: `${nom} — personnes` }), el('th', { text: `${nom} — latrines` }));
    }
    table.tHead.replaceChildren(entete);
    table.tBodies[0].replaceChildren(...b.detail.map((r) => {
      const tr = el('tr', {}, el('td', { class: 'text-start', text: String(r.annee) }), el('td', { text: nombre(r.population) }));
      for (const nom of NOMS_CAS) {
        tr.append(el('td', { text: nombre(r[`${nom}|personnes`]) }), el('td', { text: nombre(r[`${nom}|latrines`]) }));
      }
      return tr;
    }));

    $('#besoin-note').textContent =
      `Estimation : 1 latrine pour ${nombre(Number($('#personnes_par_latrine').value))} personnes (hypothèse modifiable à gauche). ` +
      'Même sans progresser, il faut construire pour suivre la croissance de la population. ' +
      'Ces chiffres sont une simulation, pas un budget ni une mesure de terrain.';
  }

  function afficherTableau(d) {
    const carte = $('#carte-tableau');
    carte.hidden = !d.tableau;
    if (!d.tableau) return;
    const cases = ['realisation', 'prevision', 'scenario', 'lineaire', 'objectif'];
    $('#table-annuelle tbody').replaceChildren(...d.tableau.map((r) =>
      el('tr', {}, el('td', { class: 'text-start', text: String(r.annee) }),
        ...cases.map((c) => el('td', { text: r[c] == null ? '—' : nombre(r[c], 2) })))));
  }

  function afficherAvertissements(d) {
    $('#alertes').replaceChildren(...d.avertissements.map((t) =>
      el('div', { class: 'alert alert-warning', role: 'status', text: t })));
  }

  function afficher(d) {
    derniere = d;
    $('#titre-territoire').textContent = d.territoire.libelle;
    $('#nb-communes').textContent = d.territoire.nb_communes > 1 ? `${nombre(d.territoire.nb_communes)} communes` : '1 commune';
    afficherIndicateurs(d);
    afficherGraphique(d);
    afficherBesoin(d);
    afficherTableau(d);
    if (!document.querySelector('#alertes .alert-danger')) afficherAvertissements(d);
  }

  // ------------------------------------------------------------------ simulation
  async function simuler() {
    if (!meta || !formulaireValide()) return;
    if (controleur) controleur.abort();            // une seule requête utile à la fois
    controleur = new AbortController();
    const signal = controleur.signal;
    $('#chargement').hidden = false;
    $('#resultats').setAttribute('aria-busy', 'true');
    try {
      const donnees = await api('/api/simulation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(lireFormulaire()),
        signal,
      });
      effacerErreur();
      afficher(donnees);
    } catch (e) {
      if (e.name !== 'AbortError') afficherErreur(e.message);
    } finally {
      if (!signal.aborted) {
        $('#chargement').hidden = true;
        $('#resultats').setAttribute('aria-busy', 'false');
      }
    }
  }

  const simulerBientot = () => { clearTimeout(minuteur); minuteur = setTimeout(simuler, 250); };

  // ------------------------------------------------------------------ onglet performance
  async function chargerPerformance() {
    const zone = $('#performance-contenu');
    try {
      const p = await api('/api/performance');
      zone.replaceChildren(...construirePerformance(p));
      performanceChargee = true;
    } catch (e) {
      zone.replaceChildren(el('div', { class: 'alert alert-info', text: e.message }));
    }
  }

  function construirePerformance(p) {
    const infos = p.infos || {};
    const noms = Object.keys(p.modeles);
    const colonnes = noms.length ? Object.keys(p.modeles[noms[0]]) : [];
    const table = el('table', { class: 'table table-sm table-striped align-middle text-end' },
      el('thead', {}, el('tr', {}, el('th', { class: 'text-start', text: 'Méthode' }), ...colonnes.map((c) => el('th', { text: c.replace('_', ' ') })))),
      el('tbody', {}, ...noms.map((n) =>
        el('tr', { class: n.includes('MLP') ? 'table-primary fw-semibold' : '' },
          el('td', { class: 'text-start', text: n }),
          ...colonnes.map((c) => el('td', { text: nombre(p.modeles[n][c], 4) }))))));

    const archi = `${(infos.variables || []).length} variables d'entrée → ${(infos.hidden_layer_sizes || []).join(' → ')} neurones ` +
      `(${infos.nb_couches_cachees ?? '?'} couches cachées, activation ${infos.activation ?? '?'}) → 1 sortie.`;

    const images = [['predit_vs_reel', 'Prédit vs réel'], ['courbes_perte', 'Courbe de perte'], ['comparaison_modeles', 'Comparaison des méthodes (erreur absolue moyenne)']]
      .filter(([cle]) => p.images[cle])
      .map(([cle, legende]) => el('div', { class: 'col-md-6' },
        el('figure', { class: 'mb-0' },
          el('img', { src: `/api/performance/image/${cle}`, alt: legende, class: 'graphique-perf border rounded bg-white', loading: 'lazy' }),
          el('figcaption', { class: 'small text-body-secondary', text: legende }))));

    const carte = (titre, classe, ...contenu) => el('div', { class: `card shadow-sm mb-3 carte-section ${classe}`.trim() },
      el('div', { class: 'card-header en-tete-section' }, el('h3', { class: 'h6 text-uppercase mb-0', text: titre })),
      el('div', { class: 'card-body' }, ...contenu));

    return [
      carte(`Évaluation sur ${infos.annee_test ?? '?'} (année jamais vue à l'entraînement)`, '',
        el('div', { class: 'table-responsive' }, table),
        el('p', { class: 'small text-body-secondary mb-0', text: 'R² mesure le niveau du taux (peu discriminant : la persistance obtient le même score) ; R² évolution mesure la progression annuelle, qui est le vrai test.' })),
      carte('Architecture du réseau', '', el('p', { class: 'mb-0', text: archi })),
      el('div', { class: 'row g-3 mb-3' }, ...images),
      carte('⚠️ Hypothèses et limites (à citer dans le mémoire)', 'carte-alerte',
        el('ul', { class: 'mb-0' }, ...p.limites.map((t) => el('li', { text: t })))),
    ];
  }

  // ------------------------------------------------------------------ démarrage
  async function demarrer() {
    try {
      meta = await api('/api/meta');
    } catch (e) {
      afficherErreur(e.message);
      $('#titre-territoire').textContent = 'Service indisponible';
      $('#chargement').hidden = true;
      return;
    }
    construireCurseursScenario();
    $('#region').replaceChildren(...meta.regions.map((r) => el('option', { value: r, text: r })));
    $('#nb_annees').value = String(meta.defauts.nb_annees);
    $('#sortie-nb_annees').textContent = String(meta.defauts.nb_annees);
    $('#objectif').value = String(meta.defauts.objectif_final);
    $('#croissance').value = String(meta.defauts.croissance_pct);
    $('#personnes_par_latrine').value = String(meta.defauts.personnes_par_latrine);
    $('#annee-cible').textContent = String(meta.annee_depart + meta.defauts.nb_annees);

    $('#type').addEventListener('change', async () => {
      majVisibiliteTerritoire();
      try { if ($('#type').value === 'commune') await chargerCommunes(); } catch (e) { afficherErreur(e.message); return; }
      simuler();
    });
    $('#region').addEventListener('change', async () => {
      try { if ($('#type').value === 'commune') await chargerCommunes(); } catch (e) { afficherErreur(e.message); return; }
      simuler();
    });
    $('#commune').addEventListener('change', simuler);
    $('#nb_annees').addEventListener('input', () => {
      $('#sortie-nb_annees').textContent = $('#nb_annees').value;
      $('#annee-cible').textContent = String(meta.annee_depart + Number($('#nb_annees').value));
      simulerBientot();
    });
    for (const id of ['#objectif', '#latrines_par_an', '#personnes_par_latrine', '#croissance']) {
      $(id).addEventListener('input', simulerBientot);
    }
    for (const c of CHAMPS_SCENARIO) c.input.addEventListener('input', simulerBientot);
    $('#formulaire').addEventListener('submit', (e) => e.preventDefault());

    $('#bascule-theme').textContent = estSombre() ? '☀️' : '🌙';
    $('#bascule-theme').addEventListener('click', basculerTheme);
    $('#onglet-performance').addEventListener('shown.bs.tab', () => { if (!performanceChargee) chargerPerformance(); });

    majVisibiliteTerritoire();
    simuler();
  }

  document.addEventListener('DOMContentLoaded', demarrer);
})();