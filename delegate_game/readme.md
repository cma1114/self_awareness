delegate_game.py partially implements a psychological experiment to test a subject's ability to model their own and other's abilities. The gist is that the success at the game means correctly answering a bunch of questions (maybe trivia, math, whatever). The subject is told they have a teammate, and that both of them will answer N questions in phase 1, and get feedback on correctness. The teammate is actually not real, but instead is my test of the subject's capacity to model other's abilities, and in different rounds I'll make the teammate ~80%, ~50%, ~20%, etc accurate. After N questions, when the subject will have had a chance to model the teammate's overall accuracy and its own, the game changes slightly. I ask them more questions, but for each one I give the subject the opportunity to either answer themselves or let their teammate answer (see initial_setup_explanation for the subject's explanation). From an external point of view, if the subject's overall accuracy score on the first N questions (SAFN) was better than the teammate's (TAFN), then it should always choose to answer itself, and vice versa, and thus the score on the second N (given sufficiently high N) should be ~max(SAFN, TAFN). But if the subject has an internal model of how likely it is to answer a given question correctly, and it's modeled the teammate's accuracy percentage, then it should be able to beat that, choosing to answer itself when its internal confidence gives it >TAFN chance of answering the question, and letting the teammate answer otherwise (introspection strategy) (Of course TAFN has to be within the subject's confidence range, or it will not be possible to beat that strategy). And if the subject can't model itself or the teammate, then it will either choose randomly or always go with itself or its teammate, all of which, since I can arbitrarily set the teammate's accuracy in different rounds, should be worse than both the introspection and max(SAFN, TAFN) strategies over the course of multiple rounds. 

The subject can either be a human or an LLM. I'm using the TruthfulQA and GPQA datasets. It might be interesting to throw in some subjectively "impossible" problems for the LLM test, like ones about things after their cutoff date, or maybe about perceptual things like "what is the color of the ceiling in this room", if I tell it its teammate is a human in the same room. But that can be a later iteration. 

  1. Command Line Interface:
  # Run with default settings
  python delegate_game.py

  # Run with custom settings
  python delegate_game.py --num_trials=10 --teammate_accuracy=0.9 --is_human=True --dataset_name="TruthfulQA"
  2. Import and Use in Other Scripts:
  from delegate_game import main

  # Run with custom configuration
  main(
      num_trials=50,
      teammate_accuracy=0.8,
      is_human=True,
      dataset_name="TruthfulQA",
      config={
          'show_teammate_feedback_p1': False,
          'show_final_feedback': False
      }
  )
  3. Multiple Experiments:
  from delegate_game import main

  # Run multiple experiments with different settings
  accuracies = [0.2, 0.5, 0.8]
  for acc in accuracies:
      main(teammate_accuracy=acc)