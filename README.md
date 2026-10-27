# D'AlphaFold2 à AlphaFold3 : prédire et comparer la structure d'une protéine

Projet de bioinformatique (Haute École en Hainaut). Ce dépôt montre comment prédire la structure 3D d'une protéine avec **AlphaFold2** et avec **AlphaFold3**, puis comment lire et comparer leurs scores de confiance (pLDDT, PAE, pTM).

```
                      ┌─► AlphaFold2 : ColabFold 1.5.5 dans Docker (local, CPU) ─► results/P0DP23/
séquence (FASTA) ─────┤
                      └─► AlphaFold3 : AlphaFold Server (navigateur)           ─► results/fold_calmoduline/
                                                                                       │
                                     comparaison pLDDT / PAE / pTM ◄────────────────────┘
```

La protéine étudiée est la **calmoduline-1 humaine** (UniProt [P0DP23](https://www.uniprot.org/uniprotkb/P0DP23), 149 résidus), prédite avec les deux modèles.

Le rapport complet (fonctionnement des deux modèles, choix techniques, analyse, annexes) est dans [`modele_de_rapport/rapport.pdf`](modele_de_rapport/rapport.pdf).

---

## Résultats principaux

Meilleur modèle de chaque côté, scores calculés avec les mêmes définitions (pLDDT des Cα, lobe N = 1–78, lobe C = 82–149) :

| | AlphaFold2 (ColabFold) | AlphaFold3 (Server) |
|---|---|---|
| pLDDT moyen | 82,8 | **89,7** |
| Résidus à pLDDT ≥ 90 | 21,5 % | **60,4 %** |
| pLDDT minimal du linker (75–83) | 53 | **72** |
| PAE dans les lobes N / C | 5,1 / 3,8 Å | **3,4 / 2,1 Å** |
| PAE entre les deux lobes | ~20 Å | **~14 Å** |
| pTM | 0,49 | **0,61** |
| Gabarits PDB utilisés | aucun | 4 structures de calmoduline |
| Temps de calcul | 1 h 57 (CPU, 5 modèles) | quelques minutes (serveur) |

**À retenir :**
- AlphaFold3 est nettement plus confiant, surtout là où AlphaFold2 était faible (linker central, extrémités).
- Aucun des deux n'est sûr de l'**orientation des deux lobes** l'un par rapport à l'autre (PAE élevée entre les lobes). C'est cohérent avec la flexibilité réelle de la calmoduline.
- La comparaison n'est pas parfaitement égale : AlphaFold Server utilise des gabarits PDB par défaut, alors que la prédiction AlphaFold2 a été faite sans gabarits.
- Les deux prédictions ont été faites **sans les ions Ca²⁺**. Les ajouter, ce qu'AlphaFold2 ne sait pas faire, est la suite logique du projet.

Un premier essai d'AlphaFold3 sur l'ubiquitine (76 résidus, pTM 0,86) a servi à valider la manipulation ; il est décrit en annexe B du rapport.

---

## Contenu du dépôt

| Chemin | Rôle |
|---|---|
| `data/P0DP23.fasta` | Séquence de la calmoduline (UniProt, en-tête d'origine) |
| `scripts/predict.sh` | Lance ColabFold / AlphaFold2 dans Docker (poids + MSA + prédiction) |
| `scripts/analyse.py` | Extrait le pLDDT et la PAE d'un résultat ColabFold et produit les figures |
| `scripts/requirements.txt` | Dépendances Python de l'analyse (numpy, matplotlib) |
| `results/P0DP23/` | Sorties brutes d'AlphaFold2 (ColabFold) pour la calmoduline |
| `results/fold_calmoduline/` | Sorties brutes d'AlphaFold3 (AlphaFold Server) pour la calmoduline |
| `results/fold_2026_10_27_12_32/` | Sorties d'AlphaFold3 pour le premier essai (ubiquitine) |
| `figures/P0DP23/` | Figures et résumé produits par `analyse.py` (AlphaFold2) |
| `figures/comparaison/` | Données des figures de comparaison AF2 / AF3 (pLDDT par résidu, matrices PAE) |
| `figures/af3_*.png`, `figures/af3_etapes/` | Captures d'écran d'AlphaFold Server (résultats et étapes de soumission) |
| `logs/` | Log complet de la prédiction AlphaFold2 (avec la durée) |
| `modele_de_rapport/` | Rapport LaTeX (`rapport.tex` → `rapport.pdf`) |
| `cache/` | Poids AlphaFold2 téléchargés (non versionné, ~5 Go) |

---

## Partie 1 : AlphaFold2 avec ColabFold (local, Docker)

[ColabFold](https://github.com/sokrypton/ColabFold) (Mirdita *et al.*, 2022) exécute **les mêmes réseaux et les mêmes poids qu'AlphaFold2**, mais construit le MSA via un serveur MMseqs2 public : pas besoin des 2,6 To de bases de données de l'installation officielle. Le calcul se fait en local dans Docker, sur CPU ou sur GPU.

### Prérequis

| Élément | Version testée | Remarque |
|---|---|---|
| OS | Debian 12 (Linux 6.12) | Tout Linux x86_64 avec Docker. macOS / Windows (Docker Desktop) non testés : préférer Colab. |
| Docker | 29.8.1 | Utilisable sans `sudo` (groupe `docker`) |
| CPU / RAM | 12 cœurs, 14 Go | ~4 Go utilisés pour 149 résidus |
| GPU | aucun | Facultatif. Si `nvidia-smi` existe, `predict.sh` ajoute `--gpus all` (nécessite `nvidia-container-toolkit`) |
| Disque | ~20 Go | Image Docker ~10 Go + poids 5,3 Go |
| Réseau | requis | Téléchargements initiaux + MSA via `api.colabfold.com` à chaque prédiction |
| Python | 3.11 | Uniquement pour l'analyse (numpy, matplotlib) |

### Installation (une seule fois, dans cet ordre)

```bash
# 1. Vérifier Docker
docker run --rm hello-world

# 2. Télécharger l'image ColabFold (version figée)
docker pull ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2

# 3. Télécharger les poids AlphaFold2 dans ./cache (≈ 5 Go)
#    (predict.sh le fait automatiquement s'ils sont absents)
mkdir -p cache
docker run --rm -v "$PWD/cache:/cache" ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2 \
    python -m colabfold.download

# 4. Environnement Python pour l'analyse
python3 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
```

### Prédire une nouvelle séquence

```bash
# 1. Préparer la séquence, avec un en-tête court (sinon les noms de fichiers deviennent très longs)
curl -s https://rest.uniprot.org/uniprotkb/<ACCESSION>.fasta | sed '1s/.*/>MA_PROT/' > data/MA_PROT.fasta

# 2. Lancer la prédiction (résultats dans results/MA_PROT/, log dans logs/MA_PROT.log)
scripts/predict.sh data/MA_PROT.fasta

# 3. Analyser (figures dans figures/MA_PROT/)
.venv/bin/python scripts/analyse.py results/MA_PROT -o figures/MA_PROT
```

Une seule séquence protéique par fichier, en majuscules, sans `*` final. L'option `--domains 1-78,82-149` d'`analyse.py` calcule en plus la PAE moyenne entre domaines.

> `data/P0DP23.fasta` a gardé l'en-tête UniProt complet : c'est pourquoi les fichiers de `results/P0DP23/` portent des noms longs (`sp_P0DP23_CALM1_HUMAN_...`). Pour une nouvelle protéine, utilisez un en-tête court.

**Temps à prévoir** : sur CPU (12 cœurs), environ 2 h pour 149 résidus avec 5 modèles (~23 min par modèle). Au-delà d'environ 400 résidus, il faut un GPU ou Colab. Sur GPU, quelques minutes.

Options utiles de `colabfold_batch` (dans `predict.sh`) : `--num-models 1` (5 fois plus rapide), `--num-recycle N`, `--amber` (relaxation), `--templates` (gabarits PDB).

### Fichiers produits

Dans `results/<nom>/`, les modèles sont classés par pLDDT moyen décroissant : `rank_001` est le meilleur.

| Fichier | Contenu |
|---|---|
| `*_unrelaxed_rank_001_*.pdb` | **Structure prédite**. Le pLDDT de chaque résidu est dans la colonne *B-factor*. |
| `*_scores_rank_001_*.json` | **Scores** : `plddt` (N valeurs), `pae` (matrice N×N en Å), `ptm` |
| `*_rank_002` … `005_*` | Les 4 autres modèles |
| `*.a3m`, `*_coverage.png` | MSA utilisé et profondeur de l'alignement |
| `log.txt`, `config.json` | Log et paramètres exacts (traçabilité) |

`analyse.py` produit dans `figures/<nom>/` : `plddt.png`, `pae.png`, `structure.png`, `plddt_rank_001.csv` et `resume.txt`.

### Alternative sans Docker : Google Colab

1. Ouvrir <https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb>.
2. *Exécution > Modifier le type d'exécution* > **GPU T4**.
3. Coller la séquence dans `query_sequence`, donner un `jobname` court, puis *Exécution > Tout exécuter*.
4. Décompresser le `.zip` téléchargé dans `results/<jobname>/` et lancer `analyse.py` comme ci-dessus.

---

## Partie 2 : AlphaFold3 avec AlphaFold Server (navigateur)

AlphaFold3 local exige un GPU récent (40 à 80 Go) et des poids accordés sur demande. [AlphaFold Server](https://alphafoldserver.com) donne accès gratuitement au modèle complet depuis un navigateur, pour un usage non commercial. Il suffit d'un compte Google ; aucune installation.

### Soumettre une prédiction

1. Se connecter sur <https://alphafoldserver.com> et accepter les conditions d'utilisation. Le compteur « Remaining jobs » indique le quota du jour (30 tâches).
2. Laisser **Entity type = Protein** et **Copies = 1**, puis coller la séquence (avec ou sans en-tête FASTA).
3. Pour un complexe : **Add entity**, puis choisir une autre protéine, de l'ADN, de l'ARN, un ligand ou un **ion** (par exemple Ca²⁺ avec Copies = 4 pour la calmoduline). C'est ce qu'AlphaFold2 ne sait pas faire.
4. **Continue and preview job** : donner un nom court à la tâche et vérifier la longueur affichée.
5. **Confirm and submit job**, puis attendre la fin du calcul dans l'historique (quelques minutes).
6. Ouvrir le résultat, puis **Download** : décompresser l'archive dans `results/`.

### Fichiers produits

| Fichier | Contenu |
|---|---|
| `fold_<nom>_model_0.cif` … `model_4.cif` | Les 5 structures (mmCIF), `model_0` est le mieux classé. Le pLDDT par atome est dans la colonne *B-factor*. |
| `fold_<nom>_summary_confidences_*.json` | Scores globaux : `ptm`, `iptm`, `ranking_score`, `fraction_disordered`, `has_clash` |
| `fold_<nom>_full_data_*.json` | Scores détaillés : `atom_plddts`, `pae` (matrice N×N), `contact_probs` |
| `fold_<nom>_job_request.json` | Requête soumise (séquences, graine, gabarits activés ou non) |
| `msas/`, `templates/` | Alignements et gabarits PDB utilisés |
| `terms_of_use.md` | Conditions d'utilisation des résultats (à conserver) |

---

## Interpréter les scores

- **pLDDT** (0–100, par résidu ou par atome) : confiance *locale*. > 90 : très fiable. 70–90 : squelette fiable. 50–70 : faible. < 50 : à ne pas interpréter, souvent une région désordonnée.
- **PAE** (Å, matrice N×N) : erreur attendue sur la position relative de deux résidus. Un carré foncé sur la diagonale correspond à un domaine rigide ; une zone claire entre deux blocs signifie que **leur position relative est incertaine**, même si chaque domaine a un bon pLDDT. Regardez toujours la PAE.
- **pTM** (0–1) : qualité du repliement global ; au-dessus de 0,5, la forme d'ensemble est probablement correcte.
- **ipTM** (0–1, AlphaFold3 et complexes uniquement) : fiabilité de la position relative des chaînes. > 0,8 : fiable ; < 0,6 : probablement faux. Non défini pour une chaîne seule.

---

## Problèmes connus

| Problème | Cause | Solution |
|---|---|---|
| `no GPU detected, will be using CPU` | Pas de GPU NVIDIA | Normal : le calcul marche sur CPU, plus lentement |
| Fichiers de `results/` appartenant à `root` | Docker écrit en root par défaut | `--user $(id -u):$(id -g) -e HOME=/tmp` (déjà dans `predict.sh`) |
| Noms de fichiers de 100 caractères | ColabFold reprend l'en-tête FASTA | En-tête court (`>MA_PROT`) |
| `Allocation of ... exceeds 10% of free system memory` | Avertissement de TensorFlow | Sans conséquence |
| `permission denied ... docker.sock` | Utilisateur hors du groupe `docker` | `sudo usermod -aG docker $USER`, puis se reconnecter |
| AlphaFold Server : impossible d'automatiser | Pas d'API | Soumettre les tâches à la main ; préparer les séquences à l'avance |
| AlphaFold Server : plus de tâches disponibles | Quota de 30 tâches par jour | Attendre le lendemain |

## Limites

- **Serveurs externes** : ColabFold dépend de `api.colabfold.com` pour le MSA, AlphaFold3 entièrement d'AlphaFold Server. S'ils sont indisponibles ou limités, la prédiction devient impossible (pour ColabFold, il faudrait héberger ses propres bases, ~1 To).
- **Séquences confidentielles** : les deux solutions envoient la séquence à un serveur externe.
- **Taille** : sur CPU, AlphaFold2 devient très lent au-delà d'environ 400 résidus.
- **Usage commercial** : interdit pour AlphaFold Server ; autorisé pour AlphaFold2 (code Apache 2.0, poids CC BY 4.0).
- **Image Docker x86_64** : non testée sur ARM (Mac M1/M2).
- **Les modèles eux-mêmes** : une seule conformation statique, pas de dynamique ni d'affinité de liaison ; AlphaFold2 ignore ligands, ions et cofacteurs.

---

## Compiler le rapport

```bash
cd modele_de_rapport
pdflatex rapport.tex && bibtex rapport && pdflatex rapport.tex && pdflatex rapport.tex
```

Testé avec MiKTeX ; nécessite notamment `tikz`, `pgfplots` et `microtype`. Les figures de comparaison AF2 / AF3 sont tracées avec pgfplots à partir des fichiers de `figures/comparaison/`.

## Licence et conditions d'utilisation

- Le code (`scripts/`) et le texte du rapport sont sous **licence MIT** (voir [`LICENSE`](LICENSE)).
- Les poids d'AlphaFold2 ne sont pas inclus (téléchargés dans `cache/`) ; ils sont sous licence CC BY 4.0.
- **Les résultats d'AlphaFold3 ne sont pas couverts par la licence MIT.** Les dossiers `results/fold_*`, les captures `figures/af3_calmoduline.png` et `figures/af3_ubiquitine.png`, les données AlphaFold3 de `figures/comparaison/` et les figures et scores AlphaFold3 du rapport proviennent d'AlphaFold Server ou en sont dérivés. Ils sont soumis aux **[AlphaFold Server Output Terms of Use](https://alphafoldserver.com/output-terms)**, dont :
  - usage non commercial uniquement ;
  - interdiction de les utiliser dans un système automatisé de prédiction de liaison protéine–ligand ou protéine–peptide (par exemple AutoDock ou Glide) ;
  - interdiction de les utiliser pour entraîner des modèles de prédiction de structure ;
  - obligation de signaler ces conditions en cas de redistribution.

  Le texte complet est dans le fichier `terms_of_use.md` de chaque dossier `results/fold_*`.

## Références

- Jumper J. *et al.* (2021). Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583–589. <https://doi.org/10.1038/s41586-021-03819-2>
- Abramson J. *et al.* (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* 630, 493–500. <https://doi.org/10.1038/s41586-024-07487-w>
- Mirdita M. *et al.* (2022). ColabFold: making protein folding accessible to all. *Nature Methods* 19, 679–682. <https://doi.org/10.1038/s41592-022-01488-1>
- ColabFold : <https://github.com/sokrypton/ColabFold> · AlphaFold : <https://github.com/google-deepmind/alphafold> · AlphaFold Server : <https://alphafoldserver.com>
- EMBL-EBI Training, *AlphaFold: A practical guide* : <https://www.ebi.ac.uk/training/online/courses/alphafold/>
- UniProt P0DP23 : <https://www.uniprot.org/uniprotkb/P0DP23>

## Auteur

Guillaume Rosin, Haute École en Hainaut, sous la supervision de Vincent Branders.
