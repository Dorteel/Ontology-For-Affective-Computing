# pip install owlready2
from owlready2 import *

# --- Ontology setup ---
onto = get_ontology("http://example.org/occ.owl#")
with onto:
    # --- Upper-level things OCC talks about ---
    class Agent(Thing): pass
    class Event(Thing): pass
    class Act(Thing): pass         # an action performed by an Agent
    class Object_(Thing): pass     # named Object_ to avoid clashing with Python's built-in

    # --- Appraisal machinery ---
    class Appraisal(Thing): pass
    class GoalRelatedAppraisal(Appraisal): pass
    class StandardRelatedAppraisal(Appraisal): pass
    class AttitudeRelatedAppraisal(Appraisal): pass
    class ProspectRelatedAppraisal(Appraisal): pass  # for prospect-based emotions (hope/fear)
    class AttributionRelatedAppraisal(Appraisal): pass  # for self/other agency, responsibility

    # Polarity / qualitative values (keep it symbolic; add numeric later if desired)
    class Desirability(GoalRelatedAppraisal): pass
    class Desirable(Desirability): pass
    class Undesirable(Desirability): pass

    class Praiseworthiness(StandardRelatedAppraisal): pass
    class Praiseworthy(Praiseworthiness): pass
    class Blameworthy(Praiseworthiness): pass

    class Appealingness(AttitudeRelatedAppraisal): pass
    class Appealing(Appealingness): pass
    class Unappealing(Appealingness): pass

    class Likelihood(ProspectRelatedAppraisal): pass
    class Likely(Likelihood): pass
    class Unlikely(Likelihood): pass

    class Agency(AttributionRelatedAppraisal): pass
    class SelfCaused(Agency): pass
    class OtherCaused(Agency): pass

    # Optional intensity drivers you can wire in later
    class Unexpectedness(Appraisal): pass
    class Effort(Appraisal): pass
    class Realization(Appraisal): pass  # event realized vs. just expected

    # --- Core OCC: emotions ---
    class Emotion(Thing): pass

    # Focus-based groupings (what is being appraised)
    class EventBasedEmotion(Emotion): pass        # appraisal of Event vs goals (desirability)
    class AgentBasedEmotion(Emotion): pass        # appraisal of Act/Agent vs standards (praiseworthiness)
    class ObjectBasedEmotion(Emotion): pass       # appraisal of Object_ vs attitudes (appealingness)
    class ProspectBasedEmotion(EventBasedEmotion): pass  # hope/fear family

    # --- Properties ---
    class aboutEvent(Emotion >> Event): pass
    class aboutAct(Emotion >> Act): pass
    class aboutObject(Emotion >> Object_): pass
    class experiencedBy(Emotion >> Agent): pass
    class hasAppraisal(Emotion >> Appraisal): pass

    # Integrity constraints (domains/ranges) for clarity
    aboutEvent.domain, aboutEvent.range = [Emotion], [Event]
    aboutAct.domain,   aboutAct.range   = [Emotion], [Act]
    aboutObject.domain,aboutObject.range= [Emotion], [Object_]
    experiencedBy.domain, experiencedBy.range = [Emotion], [Agent]
    hasAppraisal.domain, hasAppraisal.range = [Emotion], [Appraisal]

    # --- OCC families (22 canonical categories; start with the most common) ---
    # Event-based: joy / distress
    class Joy(EventBasedEmotion): pass
    class Distress(EventBasedEmotion): pass

    # Prospect-based: hope / fear; and their outcome emotions
    class Hope(ProspectBasedEmotion): pass
    class Fear(ProspectBasedEmotion): pass
    class Relief(EventBasedEmotion): pass
    class Disappointment(EventBasedEmotion): pass
    class Satisfaction(EventBasedEmotion): pass
    class FearsConfirmed(EventBasedEmotion): pass

    # Agent-based (self): pride / shame (self-caused praiseworthiness/blameworthiness)
    class Pride(AgentBasedEmotion): pass
    class Shame(AgentBasedEmotion): pass

    # Agent-based (other): admiration / reproach
    class Admiration(AgentBasedEmotion): pass
    class Reproach(AgentBasedEmotion): pass

    # Attribution blend (event desirability + agency): gratitude/anger; gratification/remorse
    class Gratitude(AgentBasedEmotion): pass
    class Anger(AgentBasedEmotion): pass
    class Gratification(AgentBasedEmotion): pass
    class Remorse(AgentBasedEmotion): pass

    # Object-based: love / hate (aka liking/disliking)
    class Love(ObjectBasedEmotion): pass
    class Hate(ObjectBasedEmotion): pass

    # --- Axiomatic patterns (lightweight, symbolic) ---
    # Joy: desirable, realized event
    Joy.equivalent_to.append(
        EventBasedEmotion
        & Restriction(hasAppraisal, SOME, Desirable)
        & Restriction(aboutEvent, SOME, Event)
    )

    # Distress: undesirable event
    Distress.equivalent_to.append(
        EventBasedEmotion
        & Restriction(hasAppraisal, SOME, Undesirable)
        & Restriction(aboutEvent, SOME, Event)
    )

    # Hope: desirable but uncertain (prospect) event
    Hope.equivalent_to.append(
        ProspectBasedEmotion
        & Restriction(hasAppraisal, SOME, Desirable)
        & Restriction(hasAppraisal, SOME, Unlikely)  # or Likely/Unlikely; tune later
        & Restriction(aboutEvent, SOME, Event)
    )

    # Fear: undesirable but possible (prospect) event
    Fear.equivalent_to.append(
        ProspectBasedEmotion
        & Restriction(hasAppraisal, SOME, Undesirable)
        & Restriction(hasAppraisal, SOME, Likely)
        & Restriction(aboutEvent, SOME, Event)
    )

    # Relief: previously feared undesirable event did NOT occur (encode via Unlikely + Realization)
    Relief.is_a.append(Restriction(hasAppraisal, SOME, Desirable))
    Disappointment.is_a.append(Restriction(hasAppraisal, SOME, Undesirable))
    Satisfaction.is_a.append(Restriction(hasAppraisal, SOME, Desirable))
    FearsConfirmed.is_a.append(Restriction(hasAppraisal, SOME, Undesirable))

    # Pride/Shame: self-caused acts judged by standards
    Pride.equivalent_to.append(
        AgentBasedEmotion
        & Restriction(hasAppraisal, SOME, Praiseworthy)
        & Restriction(hasAppraisal, SOME, SelfCaused)
        & Restriction(aboutAct, SOME, Act)
    )
    Shame.equivalent_to.append(
        AgentBasedEmotion
        & Restriction(hasAppraisal, SOME, Blameworthy)
        & Restriction(hasAppraisal, SOME, SelfCaused)
        & Restriction(aboutAct, SOME, Act)
    )

    # Admiration/Reproach: other-caused acts judged by standards
    Admiration.equivalent_to.append(
        AgentBasedEmotion
        & Restriction(hasAppraisal, SOME, Praiseworthy)
        & Restriction(hasAppraisal, SOME, OtherCaused)
        & Restriction(aboutAct, SOME, Act)
    )
    Reproach.equivalent_to.append(
        AgentBasedEmotion
        & Restriction(hasAppraisal, SOME, Blameworthy)
        & Restriction(hasAppraisal, SOME, OtherCaused)
        & Restriction(aboutAct, SOME, Act)
    )

    # Gratitude/Anger: desirable/undesirable event caused by other
    Gratitude.equivalent_to.append(
        AgentBasedEmotion
        & Restriction(hasAppraisal, SOME, Desirable)
        & Restriction(hasAppraisal, SOME, OtherCaused)
        & Restriction(aboutEvent, SOME, Event)
    )
    Anger.equivalent_to.append(
        AgentBasedEmotion
        & Restriction(hasAppraisal, SOME, Undesirable)
        & Restriction(hasAppraisal, SOME, OtherCaused)
        & Restriction(aboutEvent, SOME, Event)
    )

    # Gratification/Remorse: desirable/undesirable event caused by self
    Gratification.equivalent_to.append(
        AgentBasedEmotion
        & Restriction(hasAppraisal, SOME, Desirable)
        & Restriction(hasAppraisal, SOME, SelfCaused)
        & Restriction(aboutEvent, SOME, Event)
    )
    Remorse.equivalent_to.append(
        AgentBasedEmotion
        & Restriction(hasAppraisal, SOME, Undesirable)
        & Restriction(hasAppraisal, SOME, SelfCaused)
        & Restriction(aboutEvent, SOME, Event)
    )

    # Love/Hate: liking/disliking of object properties
    Love.equivalent_to.append(
        ObjectBasedEmotion
        & Restriction(hasAppraisal, SOME, Appealing)
        & Restriction(aboutObject, SOME, Object_)
    )
    Hate.equivalent_to.append(
        ObjectBasedEmotion
        & Restriction(hasAppraisal, SOME, Unappealing)
        & Restriction(aboutObject, SOME, Object_)
    )

# --- Minimal demo data (feel free to delete) ---
with onto:
    john = Agent("John")
    cake_party = Event("CakeParty")
    johns_help = Act("JohnHelped")
    cake = Object_("Cake")

    e1 = Joy("joy1")
    e1.experiencedBy = [john]
    e1.aboutEvent = [cake_party]
    e1.hasAppraisal = [Desirable()]  # anonymous individual is fine for a stub

    g1 = Gratitude("grat1")
    g1.experiencedBy = [john]
    g1.aboutEvent = [cake_party]
    g1.hasAppraisal = [Desirable(), OtherCaused()]

# --- Save ---
onto.save(file="occ.owl", format="rdfxml")   # or "ntriples"/"turtle"
print("Saved to occ.owl")

action_units = {
    'AU1' : 'Inner Brow Raiser',
    'AU2' : 'Outer Brow Raiser',
    'AU4' : 'Brow Lowerer',
    'AU5' : 'Upper Lid Raiser',
    'AU6' : 'Cheek Raiser',
    'AU9' : 'Nose Wrinkler',
    'AU12' : 'Lip Corner Puller',
    'AU15' : 'Lip Corner Depressor',
    'AU17' : 'Chin Raiser',
    'AU20' : 'Lip Stretcher',
    'AU25' : 'Lips Part',
    'AU26' : 'Jaw Drop'
}