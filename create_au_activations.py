from owlready2 import *
onto = get_ontology("http://example.org/facs.owl")
swrlb = onto.get_namespace("http://www.w3.org/2003/11/swrlb#")

with onto:
    class Emotion(Thing): pass
    class FaceObservation(Thing): pass
    class AU(Thing): pass
    class AUActivation(Thing): pass

    class hasActivation(FaceObservation >> AUActivation): pass
    class ofAU(AUActivation >> AU): pass
    class hasIntensity(AUActivation >> float, DataProperty): pass
    class ofCharacteristics(FaceObservation >> Emotion): pass

    # AU individuals
    # --- AU individuals (use safe names)
    au_ids = [1,2,4,5,6,7,9,10,11,12,15,16,17,20,22,23,24,25,26,27]
    AU_IND = {i: AU(f"AU_{i}") for i in au_ids}
    AU_1, AU_2, AU_4, AU_5, AU_6, AU_7, AU_9, AU_10, AU_11, AU_12, AU_15, AU_16, AU_17, AU_20, AU_22, AU_23, AU_24, AU_25, AU_26, AU_27= \
        (AU_IND[i] for i in au_ids)

    Anger = Emotion("Anger")
    Disgust = Emotion("Disgust")
    Fear = Emotion("Fear")
    Joy = Emotion("Joy")
    Sadness = Emotion("Sadness")
    Surprise = Emotion("Surprise")

    # --- ANGER ---
    # Base pattern you provided: AU4, AU5, AU7, and (23|24) × (25|26), intensities >= 2 where relevant
    facs_anger_au_rules = [[4,5,7,10,22,23,25],
                   [4,5,7,10,22,23,26],
                   [4,5,7,17,23],
                   [4,5,7,17,24],
                   [4,5,7,23],
                   [4,5,7,24],
                   [4,5],
                   [4,7],
                   [17,24]
                   ]
    
    facs_disgust_au_rules = [[9, 17],
                        [10, 17],
                        [9, 16, 25],
                        [10, 16, 25],
                        [9, 16, 26],
                        [10, 16, 26],
                        [9],
                        [10],
                        [9, 17],
                        [10, 17]
                    ]

    facs_fear_au_rules = [   [1, 2, 4],
                        [1, 2, 4, 5, 20, 25],
                        [1, 2, 4, 5, 20, 26],
                        [1, 2, 4, 5, 20, 27],
                        [1, 2, 4, 5, 25],
                        [1, 2, 4, 5, 26],
                        [1, 2, 4, 5, 27],
                        [1, 2, 4, 5],
                        [1, 2, 5, 25],
                        [1, 2, 5, 26],
                        [1, 2, 5, 27],
                        [5, 20, 25],
                        [5, 20, 26],
                        [5, 20, 27],                                                
                        [5, 20],
                        [20]
                    ]

    facs_joy_au_rules = [[12],
                        [6, 12]
                    ]
    
    facs_sadness_au_rules = [[1, 4],
                        [1, 4, 11],
                        [1, 4, 15],
                        [1, 4, 15, 17],
                        [6, 15],
                        [11, 17],
                        [1],
                    ]
    
    facs_surprise_au_rules = [[1, 2, 5, 26],
                         [1, 2, 5, 27],
                         [1, 2, 5],
                         [1, 2, 26],
                         [1, 2, 27],
                         [5, 26],
                         [5, 27]
                    ]
    
    def add_swrl_rules_for_au_emotion(rules, emotion):
        for rule in rules:
            rule_body = ""
            for au in rule:
                rule_body += f"hasActivation(?o, ?a{au}) ^ ofAU(?a{au}, AU_{au}) ^"
            rule_pattern = f"""
            FaceObservation(?o) ^
            {rule_body}
            -> ofCharacteristics(?o, {emotion})
            """
            print(rule_pattern)
            rule_var = Imp().set_as_rule(rule_pattern)
            print(rule_var),
        return rule_var
    
    add_swrl_rules_for_au_emotion(facs_anger_au_rules, 'Anger')
    add_swrl_rules_for_au_emotion(facs_disgust_au_rules, 'Disgust')
    add_swrl_rules_for_au_emotion(facs_fear_au_rules, 'Fear')
    add_swrl_rules_for_au_emotion(facs_joy_au_rules, 'Joy')
    add_swrl_rules_for_au_emotion(facs_sadness_au_rules, 'Sadness')
    add_swrl_rules_for_au_emotion(facs_surprise_au_rules, 'Surprise')



        # Example instances --------------------------------------------------
    # 1. An "angry" face (AU4, AU5, AU7 with intensity >= 2, AU23 + AU25 active)
    obs1 = FaceObservation("Obs_Anger")
    act4 = AUActivation("Act4"); ofAU[act4] = [AU_4]; hasIntensity[act4] = [3]
    act5 = AUActivation("Act5"); ofAU[act5] = [AU_5]; hasIntensity[act5] = [3]
    act7 = AUActivation("Act7"); ofAU[act7] = [AU_7]; hasIntensity[act7] = [2]
    act23 = AUActivation("Act23"); ofAU[act23] = [AU_23]
    act25 = AUActivation("Act25"); ofAU[act25] = [AU_25]
    hasActivation[obs1] = [act4, act5, act7, act23, act25]

    # 2. A "happy" face (Joy: AU6 + AU12, both intensity >= 2)
    obs2 = FaceObservation("Obs_Joy")
    act6 = AUActivation("Act6"); ofAU[act6] = [AU_6]; hasIntensity[act6] = [3]
    act12 = AUActivation("Act12"); ofAU[act12] = [AU_12]; hasIntensity[act12] = [3]
    hasActivation[obs2] = [act6, act12]

    # 3. A "disgusted" face (AU9 + AU10)
    obs3 = FaceObservation("Obs_Disgust")
    act9 = AUActivation("Act9"); ofAU[act9] = [AU_9]; hasIntensity[act9] = [3]
    act10 = AUActivation("Act10"); ofAU[act10] = [AU_10]; hasIntensity[act10] = [2]
    hasActivation[obs3] = [act9, act10]

# Save ontology
onto.save(file="facs.owl", format="rdfxml")