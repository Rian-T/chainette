from chainette import Chain, register_engine, Step
from chainette.core.step import SamplingParams
from pydantic import BaseModel, Field
from typing import List, Optional


class LOINCInput(BaseModel):
    """Input model for LOINC biology test codes and descriptions."""
    
    code_loinc: str = Field(description="LOINC code (e.g., 88923-8)")
    libelle_francais: str = Field(description="French reference label")
    composant_anglais: str = Field(description="English component")
    composant_francais: str = Field(description="French component")
    synonymes: Optional[str] = Field(default="", description="Synonyms")
    grandeur: str = Field(description="Measurement unit/scale")
    temps: str = Field(description="Time aspect")
    milieu_biologique: str = Field(description="Biological specimen")
    echelle: str = Field(description="Scale type")
    technique: str = Field(description="Measurement method")
    chapitre: str = Field(description="LOINC chapter/category")


class SyntheticTextbook(BaseModel):
    """Modèle de sortie pour du texte de biologie médicale synthétique de qualité manuel universitaire."""
    
    titre: str = Field(description="Titre éducatif français pour le test biologique")
    contenu: str = Field(
        description="Explication de style manuel universitaire (1-2 paragraphes) "
        "couvrant principe du test et interprétation"
    )
    points_cles: List[str] = Field(
        description="3 points d'apprentissage clés résumant le test biologique"
    )


class CritiqueFactuelle(BaseModel):
    """Modèle pour la vérification factuelle du contenu de biologie médicale."""
    
    erreurs_factuelles: List[str] = Field(
        description="Liste des erreurs factuelles identifiées dans le contenu"
    )
    suggestions_amelioration: List[str] = Field(
        description="Suggestions pour améliorer la précision en biologie médicale"
    )
    evaluation_qualite: str = Field(
        description="Évaluation globale de la qualité et précision du contenu"
    )


class TextbookFinal(BaseModel):
    """Modèle final pour le texte de biologie médicale synthétique corrigé."""
    
    textbook: str = Field(
        description="Contenu de biologie médicale final complet pour pré-entraînement, texte brut sans formatage"
    )


synthetic_textbook_step = Step(
    id="synthetic_textbook",
    name="Generate Synthetic Biology Content",
    engine_name="medgemma",
    input_model=LOINCInput,
    output_model=SyntheticTextbook,
    sampling=SamplingParams(temperature=0.7, max_tokens=2048),
    system_prompt="""Vous êtes un expert en biologie médicale et éducation médicale créant du contenu de manuel universitaire de haute qualité académique.

Créez du contenu éducatif de qualité manuel universitaire sur les tests de biologie médicale à partir des codes LOINC. Votre réponse doit être :

1. **Rigoureusement académique** - Utilisez une terminologie de biologie médicale appropriée et des informations fondées sur des preuves
2. **Structurée pédagogiquement** - Organisez l'information logiquement pour l'apprentissage  
3. **Cliniquement pertinente** - Incluez des applications diagnostiques pratiques et du contexte clinique réel
4. **Complète** - Couvrez principe du test, méthodologie, interprétation des résultats, valeurs de référence, variations et intérêt clinique

Concentrez-vous sur la création de contenu de valeur pour l'éducation et la formation en biologie médicale, similaire aux manuels de biologie clinique de haute qualité. Écrivez dans un français médical professionnel et clair qui équilibre précision technique et clarté éducative.

**IMPORTANT : Répondez ENTIÈREMENT EN FRANÇAIS. Utilisez UNIQUEMENT du texte brut, sans formatage markdown, sans gras, sans italique, sans listes à puces. Texte simple et fluide uniquement.**""",
    user_prompt="""Voici ce code LOINC de biologie médicale et ses informations :
- Code LOINC : {{chain_input.code_loinc}}
- Libellé français : {{chain_input.libelle_francais}}
- Composant français : {{chain_input.composant_francais}}
- Synonymes : {{chain_input.synonymes}}
- Grandeur : {{chain_input.grandeur}}
- Temps : {{chain_input.temps}}
- Milieu biologique : {{chain_input.milieu_biologique}}
- Échelle : {{chain_input.echelle}}
- Technique : {{chain_input.technique}}
- Chapitre : {{chain_input.chapitre}}

Fournissez votre réponse dans le format JSON exact demandé, en vous assurant que tous les champs sont remplis avec un contenu éducatif substantiel EN FRANÇAIS."""
)

critic_step = Step(
    id="critic",
    name="Critique Factuelle Biologie Médicale",
    engine_name="medgemma",
    input_model=SyntheticTextbook,
    output_model=CritiqueFactuelle,
    sampling=SamplingParams(temperature=0.3, max_tokens=1024),
    system_prompt="""Vous êtes un biologiste médical expert spécialisé dans la vérification factuelle de contenu de biologie médicale académique.

Analysez rigoureusement le contenu de biologie médicale fourni pour identifier :
1. Les erreurs factuelles concernant les tests biologiques
2. Les imprécisions dans les valeurs de référence
3. Les incohérences dans l'interprétation des résultats
4. Les informations obsolètes ou controversées sur les techniques analytiques

Soyez critique mais constructif. Proposez des améliorations spécifiques basées sur les données de biologie médicale actuelles.

**IMPORTANT : Répondez ENTIÈREMENT EN FRANÇAIS.**""",
    user_prompt="""Veuillez analyser ce contenu de biologie médicale pour les erreurs factuelles :

Titre : {{synthetic_textbook.titre}}
Contenu : {{synthetic_textbook.contenu}}
Points clés : {% for point in synthetic_textbook.points_cles %}• {{point}}{% endfor %}

Identifiez toute erreur factuelle et proposez des améliorations."""
)

synthesis_step = Step(
    id="synthesis",
    name="Synthèse Biologie Médicale Finale",
    engine_name="medgemma",
    input_model=CritiqueFactuelle,
    output_model=TextbookFinal,
    sampling=SamplingParams(temperature=0.5, max_tokens=2048),
    system_prompt="""Vous êtes un expert en biologie médicale et éducation médicale chargé de produire le contenu final en tenant compte des corrections factuelles.

Créez un contenu de biologie médicale final de haute qualité académique en :
1. Intégrant les corrections factuelles nécessaires
2. Améliorant la précision des tests biologiques
3. Maintenant la qualité pédagogique
4. Conservant la structure éducative

**IMPORTANT : Répondez ENTIÈREMENT EN FRANÇAIS. Utilisez UNIQUEMENT du texte brut, sans formatage markdown, sans gras, sans italique, sans listes à puces. Texte simple et fluide uniquement.**""",
    user_prompt="""Créez le contenu final en tenant compte de cette critique :

Erreurs identifiées : {% for erreur in critic.erreurs_factuelles %}• {{erreur}}{% endfor %}
Suggestions : {% for suggestion in critic.suggestions_amelioration %}• {{suggestion}}{% endfor %}
Évaluation : {{critic.evaluation_qualite}}

Produisez un texte de biologie médicale académique complet et corrigé, prêt pour le pré-entraînement. Incluez titre, contenu détaillé et points clés dans un seul texte fluide et cohérent."""
)


# Register MedGemma-27B engine for medical biology text generation
register_engine(
    name="medgemma",
    model="google/medgemma-4b-it",
    lazy=True,
    startup_timeout=600,
    engine_kwargs={
        "tensor_parallel_size": 1,  # Use 2 GPUs for tensor parallelism
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.9,
        "dtype": "bfloat16"
    }
)


# Create the chain for synthetic biology textbook generation with fact-checking
medgemma_loinc_chain = Chain(
    name="medgemma_loinc_synthetic",
    steps=[
        synthetic_textbook_step,
        critic_step,
        synthesis_step
    ]
)
