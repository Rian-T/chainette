from chainette import Chain, register_engine, Step
from chainette.core.step import SamplingParams
from pydantic import BaseModel, Field
from typing import List, Optional


class ATCInput(BaseModel):
    """Input model for ATC pharmaceutical codes and descriptions."""
    
    code: str = Field(description="ATC pharmaceutical code (e.g., L01BC09)")
    label_fr: str = Field(description="French pharmaceutical name")
    label_en: str = Field(description="English pharmaceutical name")
    parent_code: Optional[str] = Field(default="", description="Parent ATC category code")
    level: str = Field(description="Hierarchy level (1-5)")


class SyntheticTextbook(BaseModel):
    """Modèle de sortie pour du texte pharmaceutique synthétique de qualité manuel universitaire."""
    
    titre: str = Field(description="Titre éducatif français pour le médicament/classe")
    contenu: str = Field(
        description="Explication de style manuel universitaire (1-2 paragraphes) "
        "couvrant mécanisme d'action et indications principales"
    )
    points_cles: List[str] = Field(
        description="3 points d'apprentissage clés résumant le médicament/classe"
    )


class CritiqueFactuelle(BaseModel):
    """Modèle pour la vérification factuelle du contenu pharmaceutique."""
    
    erreurs_factuelles: List[str] = Field(
        description="Liste des erreurs factuelles identifiées dans le contenu"
    )
    suggestions_amelioration: List[str] = Field(
        description="Suggestions pour améliorer la précision pharmaceutique"
    )
    evaluation_qualite: str = Field(
        description="Évaluation globale de la qualité et précision du contenu"
    )


class TextbookFinal(BaseModel):
    """Modèle final pour le texte pharmaceutique synthétique corrigé."""
    
    textbook: str = Field(
        description="Contenu pharmaceutique final complet pour pré-entraînement, texte brut sans formatage"
    )


synthetic_textbook_step = Step(
    id="synthetic_textbook",
    name="Generate Synthetic Pharmaceutical Content",
    engine_name="medgemma",
    input_model=ATCInput,
    output_model=SyntheticTextbook,
    sampling=SamplingParams(temperature=0.7, max_tokens=2048),
    system_prompt="""Vous êtes un expert en pharmacologie et éducation médicale créant du contenu de manuel universitaire de haute qualité académique.

Créez du contenu éducatif de qualité manuel universitaire sur les médicaments et classes thérapeutiques à partir des codes ATC. Votre réponse doit être :

1. **Rigoureusement académique** - Utilisez une terminologie pharmaceutique appropriée et des informations fondées sur des preuves
2. **Structurée pédagogiquement** - Organisez l'information logiquement pour l'apprentissage  
3. **Cliniquement pertinente** - Incluez des applications pratiques et du contexte thérapeutique réel
4. **Complète** - Couvrez mécanisme d'action, pharmacocinétique, indications, contre-indications, effets secondaires et interactions

Concentrez-vous sur la création de contenu de valeur pour l'éducation et la formation pharmaceutiques, similaire aux manuels de pharmacologie de haute qualité. Écrivez dans un français médical professionnel et clair qui équilibre précision technique et clarté éducative.

**IMPORTANT : Répondez ENTIÈREMENT EN FRANÇAIS. Utilisez UNIQUEMENT du texte brut, sans formatage markdown, sans gras, sans italique, sans listes à puces. Texte simple et fluide uniquement.**""",
    user_prompt="""Voici ce code pharmaceutique ATC et ses informations :
- Code ATC : {{chain_input.code}}
- Nom français : {{chain_input.label_fr}}
- Nom anglais : {{chain_input.label_en}}
- Code parent : {{chain_input.parent_code}}
- Niveau hiérarchique : {{chain_input.level}}

Fournissez votre réponse dans le format JSON exact demandé, en vous assurant que tous les champs sont remplis avec un contenu éducatif substantiel EN FRANÇAIS."""
)

critic_step = Step(
    id="critic",
    name="Critique Factuelle Pharmaceutique",
    engine_name="medgemma",
    input_model=SyntheticTextbook,
    output_model=CritiqueFactuelle,
    sampling=SamplingParams(temperature=0.3, max_tokens=1024),
    system_prompt="""Vous êtes un pharmacologue expert spécialisé dans la vérification factuelle de contenu pharmaceutique académique.

Analysez rigoureusement le contenu pharmaceutique fourni pour identifier :
1. Les erreurs factuelles pharmacologiques
2. Les imprécisions concernant les mécanismes d'action
3. Les incohérences dans les indications thérapeutiques
4. Les informations obsolètes ou controversées sur les médicaments

Soyez critique mais constructif. Proposez des améliorations spécifiques basées sur les données pharmacologiques actuelles.

**IMPORTANT : Répondez ENTIÈREMENT EN FRANÇAIS.**""",
    user_prompt="""Veuillez analyser ce contenu pharmaceutique pour les erreurs factuelles :

Titre : {{synthetic_textbook.titre}}
Contenu : {{synthetic_textbook.contenu}}
Points clés : {% for point in synthetic_textbook.points_cles %}• {{point}}{% endfor %}

Identifiez toute erreur factuelle et proposez des améliorations."""
)

synthesis_step = Step(
    id="synthesis",
    name="Synthèse Pharmaceutique Finale",
    engine_name="medgemma",
    input_model=CritiqueFactuelle,
    output_model=TextbookFinal,
    sampling=SamplingParams(temperature=0.5, max_tokens=2048),
    system_prompt="""Vous êtes un expert en pharmacologie et éducation médicale chargé de produire le contenu final en tenant compte des corrections factuelles.

Créez un contenu pharmaceutique final de haute qualité académique en :
1. Intégrant les corrections factuelles nécessaires
2. Améliorant la précision pharmacologique
3. Maintenant la qualité pédagogique
4. Conservant la structure éducative

**IMPORTANT : Répondez ENTIÈREMENT EN FRANÇAIS. Utilisez UNIQUEMENT du texte brut, sans formatage markdown, sans gras, sans italique, sans listes à puces. Texte simple et fluide uniquement.**""",
    user_prompt="""Créez le contenu final en tenant compte de cette critique :

Erreurs identifiées : {% for erreur in critic.erreurs_factuelles %}• {{erreur}}{% endfor %}
Suggestions : {% for suggestion in critic.suggestions_amelioration %}• {{suggestion}}{% endfor %}
Évaluation : {{critic.evaluation_qualite}}

Produisez un texte pharmaceutique académique complet et corrigé, prêt pour le pré-entraînement. Incluez titre, contenu détaillé et points clés dans un seul texte fluide et cohérent."""
)


# Register MedGemma-27B engine for pharmaceutical text generation
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


# Create the chain for synthetic pharmaceutical textbook generation with fact-checking
medgemma_atc_chain = Chain(
    name="medgemma_atc_synthetic",
    steps=[
        synthetic_textbook_step,
        critic_step,
        synthesis_step
    ]
)
