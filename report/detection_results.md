# Détection de mèmes haineux : exploration des stratégies de prompting

## 3.1 Motivation : les limites du prompting zéro-shot

La détection automatique de mèmes haineux est une tâche particulièrement difficile pour les modèles de vision et de langage (VLM). Contrairement aux textes haineux conventionnels, un mème peut être inoffensif lorsque son image et son texte sont considérés séparément, mais devenir profondément problématique par leur combinaison. Un mème qui superpose une légende anodine à une image raciste, ou une image neutre à un texte de propagande, ne peut être analysé qu'à la lumière de cet effet de composition multimodal.

Dans ce contexte, le prompting zéro-shot — c'est-à-dire fournir uniquement une définition du discours haineux et demander au modèle de classer le mème — présente plusieurs lacunes structurelles. Premièrement, sans exemples de calibration, les modèles tendent à produire des probabilités mal calibrées : des mèmes manifestement haineux obtiennent des scores inférieurs au seuil de 0,5, non par manque de compréhension sémantique, mais parce que le modèle n'a pas de repère pour ancrer l'intensité de ses prédictions. Deuxièmement, la haine implicite — celle qui passe par l'ironie, le code culturel ou la référence historique — exige un guidage interprétatif que le seul énoncé d'une définition ne suffit pas à apporter. Troisièmement, des registres rhétoriques comme l'humour, le sarcasme ou le langage motivationnel peuvent masquer une intention haineuse ou, inversement, amener le modèle à classer comme haineux un contenu qui est simplement provocateur ou vulgaire sans cibler un groupe protégé.

Ces constats ont motivé l'exploration de six stratégies de prompting complémentaires, décrites ci-dessous, conçues pour enrichir le raisonnement du modèle selon différentes dimensions : calibration par l'exemple, conditionnement affectif, et routage thématique.

---

## 3.2 Description des six pipelines testés

Les six pipelines ont tous été évalués avec le même modèle multimodal — Qwen3.6-27B — et sur le même jeu d'évaluation (490 mèmes équilibrés, 245 haineux / 245 non-haineux, issus du Hateful Memes Challenge). Ils se distinguent par le nombre d'appels au modèle et par le type d'information fournie à chaque étape.

### 3.2.1 `fewshot_synthetic` — few-shot avec exemples synthétiques

Ce pipeline repose sur un unique appel VLM. Le prompt intègre la définition du discours haineux, les critères de classification, et dix-sept exemples de calibration synthétiques couvrant un large spectre de types de haine : explicite (insultes, appels à la violence), implicite (humour normalisateur, stéréotypes pseudo-scientifiques), culturellement codée (symboles suprémacistes, codes numériques d'extrême droite), historique (propagande nazie recyclée, symbolique confédérale) et intersectionnelle (attaques combinant race, religion et genre).

Ces exemples ont été rédigés spécifiquement pour corriger le problème de calibration observé en zéro-shot : chaque exemple illustre le type de haine en cause, explique pourquoi le contenu est hateux, et indique l'ordre de grandeur de probabilité attendu. L'objectif est de permettre au modèle de se positionner sur l'échelle de probabilité de façon cohérente avec les exemples fournis.

### 3.2.2 `fewshot_real` — few-shot avec exemples réels annotés

Ce pipeline est structurellement similaire au précédent (un seul appel VLM), mais les exemples de calibration sont des instances réelles extraites du jeu de données Hateful Memes, avec leur texte original et leur étiquette officielle. Quatre exemples ont été sélectionnés pour couvrir les catégories historique, identitaire et culturelle. Ces exemples sont exclus du calcul des métriques d'évaluation afin d'éviter toute contamination.

L'idée centrale est que des exemples tirés du domaine cible — plutôt que synthétiques — fournissent une ancre de calibration plus fidèle à la distribution réelle des données. La contrepartie est que les exemples réels sont moins contrôlables dans leur couverture typologique et que leur sélection peut introduire un biais vers certains types de haine.

### 3.2.3 `sentiment_single` — conditionnement affectif en un seul appel

Ce pipeline demande au modèle, dans un unique prompt, d'analyser explicitement cinq dimensions affectives avant de rendre sa classification. Ces dimensions — détaillées à la section 3.3 — incluent le sentiment global, l'humour, le sarcasme, le caractère offensant et le caractère motivationnel. Le prompt contient les définitions précises de chaque dimension et des règles d'interprétation explicites (par exemple : « l'humour seul n'est pas une preuve de haine ; évaluez si la blague cible un groupe protégé »).

L'hypothèse derrière ce pipeline est que rendre le raisonnement affectif explicite, mais en une seule passe, permet au modèle de mieux distinguer un contenu offensant ou sarcastique d'un contenu réellement haineux, sans introduire la complexité d'une architecture en chaîne.

### 3.2.4 `sentiment_chained` — conditionnement affectif en chaîne de deux appels

Ce pipeline décompose la tâche en deux appels VLM successifs. Le premier appel est dédié exclusivement à la classification affective : le modèle produit un vecteur de labels (sentiment, humour, sarcasme, degré d'offense, caractère motivationnel) pour le mème, sans porter de jugement sur sa haineuseté. Le second appel reçoit ces labels détectés, les définitions correspondantes et des règles d'interprétation adaptées à la combinaison de labels observée, puis classe le mème comme haineux ou non.

L'idée est que séparer la reconnaissance du registre communicatif de la décision de haineuseté devrait permettre une analyse plus fine. En théorie, connaître qu'un mème est « très sarcastique » et « légèrement offensant » permet d'ajuster les critères d'évaluation de haineuseté en conséquence. En pratique, cette architecture suppose que la classification affective du premier appel est fiable et que les erreurs ne se propagent pas au second appel.

### 3.2.5 `category_sentiment` — routage par catégorie suivi d'un conditionnement affectif

Ce pipeline introduit un niveau de routage thématique préalable. Un premier appel classe le mème dans l'une de trois catégories : `historical` (référence à des événements historiques, génocides, figures politiques), `general_culture` (vie quotidienne, culture populaire, humour généraliste) ou `identity_social` (contenu ciblant un groupe identitaire en contexte contemporain — race, religion, genre, orientation sexuelle, etc.).

Pour les mèmes `historical`, un second appel utilise directement des exemples few-shot ciblant les formes de haine historique (propagande nazie, iconographie de l'esclavagisme, négationnisme). Pour les autres catégories, un second appel effectue la classification affective, puis un troisième appel combine la catégorie détectée, les labels affectifs et des règles d'interprétation spécifiques à la catégorie pour produire la décision finale.

Ce pipeline est le plus complexe des six : il peut impliquer jusqu'à trois appels successifs, ce qui soulève la question de la propagation des erreurs à travers la chaîne.

### 3.2.6 `category_fewshot` — routage par catégorie suivi d'exemples few-shot ciblés

Ce pipeline conserve la première étape de routage thématique de `category_sentiment`, mais remplace entièrement l'étape de conditionnement affectif par des exemples few-shot spécifiques à la catégorie détectée. Pour chacune des trois catégories, deux exemples annotés du jeu de données sont fournis au second appel, accompagnés d'une explication des mécanismes de haine en jeu dans ces exemples.

L'architecture est donc plus simple que `category_sentiment` (deux appels au lieu de trois) et évite la difficulté de l'interprétation affective intermédiaire. Elle repose sur l'hypothèse que la combinaison d'un contexte catégoriel et d'exemples ciblés est suffisante pour guider la décision de haineuseté, sans nécessiter une modélisation explicite du registre communicatif.

---

## 3.3 Dimensions affectives issues du jeu de données Memotion

Les pipelines `sentiment_single`, `sentiment_chained` et `category_sentiment` s'appuient sur un ensemble de cinq dimensions affectives inspirées du cadre de la tâche SemEval-2020 Task 8 — Memotion Analysis (Sharma et al., 2020). Ces dimensions ont été opérationnalisées dans les prompts comme suit :

**Sentiment global** (*overall sentiment*) : polarité affective générale du mème — positive, neutre ou négative. Cette dimension capture le ton émotionnel d'ensemble, indépendamment de toute considération de haineuseté. Un mème au sentiment négatif n'est pas nécessairement haineux, et un mème au sentiment positif ou neutre peut l'être s'il attaque, déshumanise ou exclut un groupe protégé.

**Humour** : mode communicatif dans lequel une transgression, une incongruité ou une violation de norme est présentée comme bénigne, acceptable ou comique. L'humour est gradué sur quatre niveaux (`not_funny`, `funny`, `very_funny`, `hilarious`). L'humour seul ne constitue pas une preuve de haineuseté ; la question pertinente est de savoir si la blague dépend d'une attaque, d'une humiliation ou d'une déshumanisation de personnes sur la base de caractéristiques protégées.

**Sarcasme** : mode communicatif dans lequel le sens intentionnel peut différer du sens littéral, ou même lui être opposé. Quatre niveaux sont distingués (`not_sarcastic`, `little_sarcastic`, `very_sarcastic`, `extremely_sarcastic`). Lors de l'évaluation de la haineuseté, le modèle est explicitement guidé pour ne pas s'appuyer uniquement sur le sens littéral, mais pour évaluer le sens implicite, la cible implicite et si le message sarcastique attaque un groupe protégé.

**Caractère offensant** (*offense*) : contenu susceptible d'être vulgaire, insultant, profane ou socialement inapproprié, gradué sur quatre niveaux (`not_offensive`, `slight_offensive`, `very_offensive`, `hateful_offensive`). Les prompts distinguent explicitement l'offensivité générale de la haineuseté : un contenu peut être offensant sans cibler de groupe protégé, et inversement, un contenu peut être haineux sans être formellement vulgaire.

**Caractère motivationnel** (*motivation*) : contenu visant à encourager, inspirer ou promouvoir une attitude ou un comportement, classé binaiement (`not_motivational` / `motivational`). Le cadrage motivationnel n'est pas haineux en soi ; il le devient si le message encourage la supériorité, l'exclusion ou la nuisance envers des personnes sur la base de caractéristiques protégées.

> Sharma, C., Bhageria, D., Scott, W., Pykl, S., Das, A., Chakraborty, T., Pulabaigari, V., & Gamback, B. (2020). SemEval-2020 Task 8: Memotion Analysis — The Visuo-Lingual Metaphor! In *Proceedings of the Fourteenth Workshop on Semantic Evaluation* (SemEval-2020), pp. 759–773. International Committee for Computational Linguistics.

---

## 3.4 Résultats expérimentaux et comparaison des pipelines

### 3.4.1 Métriques principales

Le tableau suivant présente les résultats de chacun des six pipelines sur le jeu d'évaluation équilibré (490 mèmes). Les métriques de précision, rappel et F1 sont calculées pour la classe haineuse (`label = 1`).

| Pipeline | AUROC | Macro-F1 | Accuracy | Précision (hain.) | Rappel (hain.) | F1 (hain.) |
|---|---|---|---|---|---|---|
| `fewshot_synthetic` | **0,7552** | **0,7102** | **0,7102** | 0,7102 | 0,7102 | 0,7102 |
| `fewshot_real` | 0,7493 | 0,6723 | 0,6796 | 0,6384 | 0,8286 | 0,7211 |
| `category_fewshot` | 0,7208 | 0,6621 | 0,6735 | 0,6269 | **0,8571** | **0,7241** |
| `category_sentiment` | 0,6609 | 0,6459 | 0,6469 | 0,6324 | 0,7020 | 0,6654 |
| `sentiment_single` | 0,6475 | 0,6244 | 0,6245 | 0,6215 | 0,6367 | 0,6290 |
| `sentiment_chained` | 0,6344 | 0,6116 | 0,6122 | 0,6038 | 0,6531 | 0,6275 |

### 3.4.2 Matrices de confusion

| Pipeline | VP | VN | FP | FN | FP (%) | FN (%) |
|---|---|---|---|---|---|---|
| `fewshot_synthetic` | 174 | 174 | 71 | 71 | 29,0 | 29,0 |
| `fewshot_real` | 203 | 130 | 115 | 42 | 46,9 | 17,1 |
| `category_fewshot` | 210 | 120 | 125 | 35 | 51,0 | **14,3** |
| `category_sentiment` | 172 | 145 | 100 | 73 | 40,8 | 29,8 |
| `sentiment_single` | 156 | 150 | 95 | 89 | 38,8 | 36,3 |
| `sentiment_chained` | 160 | 140 | 105 | 85 | 42,9 | 34,7 |

### 3.4.3 Analyse

**`fewshot_synthetic` présente les meilleures performances globales.** Ce pipeline remporte les trois métriques agrégées (AUROC 0,7552, Macro-F1 0,7102, accuracy 0,7102) et se distingue par un comportement de classification parfaitement symétrique : 71 faux positifs et 71 faux négatifs, soit exactement 29 % d'erreur dans chaque direction. Cette symétrie reflète un seuil de décision bien calibré, sans biais systématique vers l'une ou l'autre classe.

**`fewshot_real` et `category_fewshot` privilégient le rappel au détriment de la précision.** Ces deux pipelines manquent très peu de mèmes haineux (respectivement 17,1 % et 14,3 % de faux négatifs), mais signalent à tort une proportion importante de mèmes non-haineux (46,9 % et 51,0 % de faux positifs). Ce déséquilibre s'explique : les exemples fournis dans ces deux pipelines illustrent presque exclusivement des cas haineux, ce qui déplace le seuil implicite du modèle vers une plus grande vigilance. Pour `fewshot_real`, les quatre exemples sont tous haineux ; pour `category_fewshot`, chaque catégorie dispose de deux exemples, tous haineux également.

**Les pipelines de conditionnement affectif sous-performent sur l'ensemble des métriques.** `sentiment_single`, `sentiment_chained` et `category_sentiment` obtiennent tous un AUROC inférieur à 0,67 et un Macro-F1 inférieur à 0,65. L'analyse de la littérature suggère que le conditionnement affectif est le plus utile lorsque le modèle a tendance à confondre le registre communicatif avec la haineuseté — par exemple, classer comme haineux un mème sarcastique mais inoffensif. Or, Qwen3.6-27B semble déjà gérer une partie de cette distinction en interne, de sorte que les labels affectifs intermédiaires ajoutent du bruit plutôt que du signal.

**`sentiment_chained` est le pipeline le moins performant malgré sa complexité.** Son AUROC de 0,6344 est le plus bas de l'ensemble, ce qui illustre un phénomène de dégradation en cascade : les erreurs de la première étape de classification affective se propagent à la seconde étape de détection de la haine, qui reçoit des labels erronés et construit un prompt de détection inapproprié. Plus la chaîne d'inférence est longue, plus le risque d'accumulation d'erreurs intermédiaires est élevé.

**Le routage thématique seul (`category_fewshot`) récupère l'essentiel du gain des exemples few-shot** sans l'étape de conditionnement affectif, ce qui confirme que le signal utile dans `category_sentiment` provient principalement du contexte catégoriel et non des labels d'affect. En d'autres termes, savoir qu'un mème est de type `identity_social` permet déjà d'appliquer un niveau de scrutin plus élevé, indépendamment de son registre communicatif.

---

## 3.5 Choix de `category_fewshot` comme pipeline de référence

À la lecture des résultats, `fewshot_synthetic` obtient les meilleures performances globales sur les métriques agrégées. Pourquoi alors retenir `category_fewshot` comme pipeline de référence pour la suite du système ?

Le choix s'explique par une combinaison de considérations relatives à la tâche applicative, aux métriques de détection et à l'interprétabilité du système.

**Du point de vue de la tâche applicative.** Le pipeline de détection est intégré dans un système de modération automatique dont l'objectif est de neutraliser les mèmes haineux avant leur diffusion. Dans ce cadre, l'asymétrie des erreurs est fondamentale : un faux négatif (un mème haineux non détecté) laisse passer du contenu préjudiciable sans traitement, tandis qu'un faux positif (un mème non-haineux signalé à tort) soumet un contenu inoffensif à une étape de neutralisation par diffusion. Cette dernière erreur est moins grave sur le plan éthique et social. `category_fewshot` minimise les faux négatifs à 14,3 % (35 mèmes manqués sur 245), contre 29,0 % pour `fewshot_synthetic` (71 mèmes manqués). Ce gain de rappel de quinze points de pourcentage représente trente-six mèmes haineux supplémentaires correctement interceptés.

**Du point de vue des métriques de détection.** Si `fewshot_synthetic` domine sur l'AUROC et le Macro-F1, `category_fewshot` obtient le meilleur F1 sur la classe haineuse (0,7241 contre 0,7102), ce qui en fait le pipeline le plus performant sur la tâche principale qui nous intéresse. Le Macro-F1 plus faible de `category_fewshot` reflète sa précision plus basse sur les non-haineux (davantage de faux positifs), mais ce compromis est acceptable dans notre cas d'usage. Par ailleurs, l'AUROC de 0,7208 reste compétitif : il indique que le modèle discrimine bien les deux classes en termes de scores continus, même si le point de fonctionnement optimal est décalé vers un rappel plus élevé.

**Du point de vue de l'interprétabilité et de la cohérence du système.** Le routage thématique produit une information structurée — la catégorie du mème — qui est exploitable pour l'analyse des erreurs et pour l'amélioration du système. Un mème mal classé dans une catégorie peut être audité, et les exemples few-shot peuvent être affinés catégorie par catégorie. Par contraste, le comportement de `fewshot_synthetic` est monolithique : tous les mèmes traversent le même prompt de 17 exemples, sans distinction contextuelle. La catégorisation offre également une légère protection contre les faux positifs dans les cas ambigus : un mème de culture générale est analysé avec des exemples et des règles moins sévères qu'un mème identitaire ou historique, ce qui réduit le risque de surévaluation du degré de haineuseté pour des contenus satiriques ou humoristiques sans cible protégée.

**Du point de vue de l'architecture.** `category_fewshot` n'implique que deux appels VLM, ce qui le rend nettement plus léger que `category_sentiment` (jusqu'à trois appels). Il évite les problèmes de propagation d'erreurs liés à la chaîne affective, tout en conservant la capacité de contextualisation apportée par le routage thématique.

En résumé, `category_fewshot` constitue le meilleur compromis entre rappel élevé sur la classe haineuse, interprétabilité du pipeline, simplicité architecturale et performances compétitives sur l'ensemble des métriques. C'est pour ces raisons qu'il a été retenu comme pipeline de détection de référence dans le système final.
