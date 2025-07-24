from chainette import Chain, register_engine, Step
from chainette.core.step import SamplingParams
from pydantic import BaseModel, Field
from typing import List, Optional


class CIM10Input(BaseModel):
    """Input model for CIM-10 medical codes and descriptions."""
    
    code: str = Field(description="CIM-10 medical code (e.g., F02.00)")
    label: str = Field(description="Medical condition label in French")
    description: Optional[str] = Field(default="", description="Additional description")
    note: Optional[str] = Field(default="", description="Additional notes")
    inclusion_notes: List[str] = Field(default_factory=list, description="Inclusion criteria")
    exclusion_notes: List[str] = Field(default_factory=list, description="Exclusion criteria")
    parent_label: Optional[str] = Field(default="", description="Parent category label")
    full_description: Optional[str] = Field(default="", description="Complete formatted description")


class SyntheticTextbook(BaseModel):
    """Modèle de sortie pour du texte médical synthétique de qualité manuel universitaire."""
    
    titre: str = Field(description="Titre éducatif français pour la condition médicale")
    contenu: str = Field(
        description="Explication de style manuel universitaire (1-2 paragraphes) "
        "couvrant définition et présentation clinique"
    )
    points_cles: List[str] = Field(
        description="3 points d'apprentissage clés résumant la condition"
    )
    example_clinique: str = Field(description="Un cas clinique synthétique"
            "typique et réaliste tel que dans les hôpitaux français representant correctement ce code"
    )


class CritiqueFactuelle(BaseModel):
    """Modèle pour la vérification factuelle du contenu médical."""
    
    erreurs_factuelles: List[str] = Field(
        description="Liste des erreurs factuelles identifiées dans le contenu"
    )
    suggestions_amelioration: List[str] = Field(
        description="Suggestions pour améliorer la précision médicale"
    )
    evaluation_qualite: str = Field(
        description="Évaluation globale de la qualité et précision du contenu"
    )


class TextbookFinal(BaseModel):
    """Modèle final pour le texte médical synthétique corrigé."""
    
    textbook: str = Field(
        description="Contenu médical final complet pour pré-entraînement, texte brut sans formatage"
    )


synthetic_textbook_step = Step(
    id="synthetic_textbook",
    name="Generate Synthetic Textbook Content",
    engine_name="medgemma",
    input_model=CIM10Input,
    output_model=SyntheticTextbook,
    sampling=SamplingParams(temperature=0.3, max_tokens=8048),
    system_prompt="""Vous êtes un expert en éducation médicale créant du contenu de manuel universitaire de haute qualité académique.

Créez du contenu éducatif de qualité manuel universitaire sur les conditions médicales à partir des codes CIM-10. Votre réponse doit être :

1. **Rigoureusement académique** - Utilisez une terminologie médicale appropriée et des informations fondées sur des preuves
2. **Structurée pédagogiquement** - Organisez l'information logiquement pour l'apprentissage  
3. **Cliniquement pertinente** - Incluez des applications pratiques et du contexte réel
4. **Complète** - Couvrez définition, physiopathologie, présentation, diagnostic et prise en charge

Concentrez-vous sur la création de contenu de valeur pour l'éducation et la formation médicales, similaire aux manuels médicaux de haute qualité. Écrivez dans un français médical professionnel et clair qui équilibre précision technique et clarté éducative.

**IMPORTANT : Répondez ENTIÈREMENT EN FRANÇAIS. Utilisez UNIQUEMENT du texte brut, sans formatage markdown, sans gras, sans italique, sans listes à puces. Texte simple et fluide uniquement.**""",
    user_prompt="""Voici ce code médical CIM-10 et ses informations :
- Code : {{chain_input.code}}
- Condition : {{chain_input.label}}
- Catégorie parente : {{chain_input.parent_label}}
- Description complète : {{chain_input.full_description}}
{% if chain_input.inclusion_notes %}
- Notes d'inclusion : {% for note in chain_input.inclusion_notes %}• {{note}}{% endfor %}
{% endif %}
{% if chain_input.exclusion_notes %}
- Notes d'exclusion : {% for note in chain_input.exclusion_notes %}• {{note}}{% endfor %}
{% endif %}

Fournissez votre réponse dans le format JSON exact demandé, en vous assurant que tous les champs sont remplis avec un contenu éducatif substantiel EN FRANÇAIS."""
)

critic_step = Step(
    id="critic",
    name="Critique Factuelle",
    engine_name="medgemma",
    input_model=SyntheticTextbook,
    output_model=CritiqueFactuelle,
    sampling=SamplingParams(temperature=0.0, max_tokens=4024),
    system_prompt="""Vous êtes un médecin expert spécialisé dans la vérification factuelle de contenu médical académique.

Analysez rigoureusement le contenu médical fourni pour identifier :
1. Les erreurs factuelles médicales
2. Les imprécisions terminologiques 
3. Les incohérences cliniques
4. Les informations obsolètes ou controversées

Soyez critique mais constructif. Proposez des améliorations spécifiques basées sur les données médicales actuelles.

**IMPORTANT : Répondez ENTIÈREMENT EN FRANÇAIS.**""",
    user_prompt="""Veuillez analyser ce contenu médical pour les erreurs factuelles :

Titre : {{synthetic_textbook.titre}}
Contenu : {{synthetic_textbook.contenu}}
Cas: {{synthetic_textbook.example_clinique}}
Points clés : {% for point in synthetic_textbook.points_cles %}• {{point}}{% endfor %}

Identifiez toute erreur factuelle et proposez des améliorations."""
)

synthesis_step = Step(
    id="synthesis",
    name="Synthèse Finale",
    engine_name="medgemma",
    input_model=CritiqueFactuelle,
    output_model=TextbookFinal,
    sampling=SamplingParams(temperature=0.1, max_tokens=8048),
    system_prompt="""Vous êtes un expert en éducation médicale chargé de produire le contenu final en tenant compte des corrections factuelles.

Créez un contenu médical final de haute qualité académique en :
1. Intégrant les corrections factuelles nécessaires
2. Améliorant la précision médicale
3. Maintenant la qualité pédagogique
4. Conservant la structure éducative

**IMPORTANT : Répondez ENTIÈREMENT EN FRANÇAIS. Utilisez UNIQUEMENT du texte brut, sans formatage markdown, sans gras, sans italique, sans listes à puces. Texte simple et fluide uniquement.**""",
    user_prompt="""Créez le contenu final en tenant compte de cette critique :

Erreurs identifiées : {% for erreur in critic.erreurs_factuelles %}• {{erreur}}{% endfor %}
Suggestions : {% for suggestion in critic.suggestions_amelioration %}• {{suggestion}}{% endfor %}
Évaluation : {{critic.evaluation_qualite}}

Voici le contenu sur lequel se baser en exploitant la correction:
Titre : {{synthetic_textbook.titre}}
Contenu : {{synthetic_textbook.contenu}}
Cas: {{synthetic_textbook.example_clinique}}
Points clés : {% for point in synthetic_textbook.points_cles %}• {{point}}{% endfor %}

Produisez un texte médical académique complet et corrigé, prêt pour le pré-entraînement. Incluez titre, contenu détaillé et points clés dans un seul texte fluide et cohérent."""
)


# Register MedGemma-27B engine for medical text generation
register_engine(
    name="medgemma",
    model="google/medgemma-27b-text-it",  # Using the correct 27B instruction-tuned model
    lazy=True,
    startup_timeout=900,  # 15 minutes for large model download/loading/compilation
    # vLLM parameters for 27B model with 2 GPUs
    tensor_parallel_size=2,  # Use 2 GPUs for tensor parallelism
    max_model_len=8192,      # Reasonable context length
    gpu_memory_utilization=0.9,
    dtype="bfloat16"          # Use bfloat16 for memory efficiency
)


# Create the chain for synthetic textbook generation with fact-checking
medgemma_cim10_chain = Chain(
    name="medgemma_cim10_synthetic",
    steps=[
        synthetic_textbook_step,
        critic_step,
        synthesis_step
    ]
)
