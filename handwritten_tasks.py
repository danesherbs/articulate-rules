import rules
from datasets import (
    DEFAULT_PROMPT_INSTRUCTION,
    BinaryClassificationTask,
    BinaryClassificationExample,
    FreeformExplanationTask,
)

########################
# CLASSIFICATION TASKS #
########################


HANDWRITTEN_CLASSIFICATION_TASKS_WITH_CHAIN_OF_THOUGHT = {
    rules.words_are_in_reverse_alphabetical_order: BinaryClassificationTask(
        instruction=(
            "The following inputs are labelled 'True' if they match a pattern and"
            " 'False' otherwise. The pattern is known to be very simple and explainable"
            " in plain English. Label the remaining inputs according to the pattern."
        ),
        examples=(
            BinaryClassificationExample(
                value="ayne ales cation status dict", label=False
            ),
            BinaryClassificationExample(
                value="hyper shut button pared said", label=False
            ),
            BinaryClassificationExample(
                value="rf ranged fab enforcement cling", label=True
            ),
            BinaryClassificationExample(
                value="boost coe values ems antes", label=False
            ),
            BinaryClassificationExample(
                value="ream pathic happy graduate broad", label=True
            ),
            BinaryClassificationExample(
                value="might rounded gallery numbered peed", label=False
            ),
            BinaryClassificationExample(
                value="worthy front cair vision ending", label=False
            ),
            BinaryClassificationExample(
                value="compliance request unsigned adays ohm", label=False
            ),
            BinaryClassificationExample(
                value="recorded prises ousel neutral dx", label=True
            ),
            BinaryClassificationExample(
                value="raped genre ching failed paces", label=False
            ),
            BinaryClassificationExample(
                value="tun mask ridden police hey", label=False
            ),
            BinaryClassificationExample(
                value="voc ravel raining impact cause", label=True
            ),
            BinaryClassificationExample(
                value="serving same raved god dump", label=True
            ),
            BinaryClassificationExample(
                value="udder reported ored ills boost", label=True
            ),
            BinaryClassificationExample(
                value="visible slice serious qt filter", label=True
            ),
            BinaryClassificationExample(
                value="when wake timer speaking lp", label=True
            ),
            BinaryClassificationExample(
                value="points nostic hots bidden basic", label=True
            ),
            BinaryClassificationExample(
                value="operator andy seed boot vell", label=False
            ),
            BinaryClassificationExample(
                value="stract shore mt izzard holding", label=True
            ),
            BinaryClassificationExample(value="eely wl aspers trop ails", label=False),
            BinaryClassificationExample(
                value="robe grades fan different demon", label=True
            ),
            BinaryClassificationExample(
                value="shown pend imposed fax abby", label=True
            ),
            BinaryClassificationExample(
                value="ounding automatic better trigger character", label=False
            ),
            BinaryClassificationExample(
                value="instead pty filter cakes seeing", label=False
            ),
            BinaryClassificationExample(
                value="uberty lins visible professional gamer", label=False
            ),
            BinaryClassificationExample(
                value="aval ends avour suggest hao", label=False
            ),
            BinaryClassificationExample(
                value="rd font birth asses adjusted", label=True
            ),
            BinaryClassificationExample(
                value="lain byss amount shadow ahead", label=False
            ),
            BinaryClassificationExample(
                value="worldly vim iced dirty associated", label=True
            ),
            BinaryClassificationExample(value="sed nine lf holm bike", label=True),
            BinaryClassificationExample(
                value="oration ridges crew values topic", label=False
            ),
            BinaryClassificationExample(
                value="working herry coins bold axis",
                label=True,
            ),
        ),
        chain_of_thought=(
            "The passing examples 'voc ravel raining impact cause' and 'rd font birth"
            " asses adjusted' don't have any words in common. This suggests that the"
            " pattern doesn't depend on whether or not a certain word appears in the"
            " input. The failing examples 'lain byss amount shadow ahead' and 'operator"
            " andy seed boot vell' also don't have any words in common This is also"
            " consistent with a pattern that doesn't depend on whether or not a certain"
            " word appears in the input. The passing example 'ream pathic happy"
            " graduate broad' starts with the word 'ream' and ends with the word"
            " 'broad'. The failing example 'aval ends avour suggest hao' starts with"
            " the word 'aval' and ends with the word 'hao'. The word 'ream' comes after"
            " the word 'broad' in the alphabet and is a passing example. The word"
            " 'aval' comes before the word 'hao' in the alphabet and is a failing"
            " example. This suggests that the pattern depends on the order of the words"
            " in the input. The words in the passing example 'ream pathic happy"
            " graduate broad' are in reverse alphabetical order. The words in the"
            " failing example 'aval ends avour suggest hao' are not in reverse"
            " alphabetical order. Therefore the pattern is that the words in the input"
            " must be in reverse alphabetical order. Therefore the label for 'working"
            " herry coins bold axis' is `True`."
        ),
    ),
    rules.words_are_in_alphabetical_order: BinaryClassificationTask(
        instruction=(
            "The following inputs are labelled 'True' if they match a pattern and"
            " 'False' otherwise. The pattern is known to be very simple and explainable"
            " in plain English. Label the remaining inputs according to the pattern."
        ),
        examples=(
            BinaryClassificationExample(
                value="hend ville simple scroll roots", label=False
            ),
            BinaryClassificationExample(
                value="allowed enforcement natal perse protein", label=True
            ),
            BinaryClassificationExample(
                value="abilities forums nit utch wich", label=True
            ),
            BinaryClassificationExample(
                value="ese adays dragon raid rays", label=False
            ),
            BinaryClassificationExample(
                value="shadow motion agree erd agency", label=False
            ),
            BinaryClassificationExample(
                value="along bors estamp home scientific", label=True
            ),
            BinaryClassificationExample(
                value="iw okia frequency testing rax", label=False
            ),
            BinaryClassificationExample(value="ahead ems grab mental olds", label=True),
            BinaryClassificationExample(value="want gross odic vice hub", label=False),
            BinaryClassificationExample(
                value="being laughter nav pole thought", label=True
            ),
            BinaryClassificationExample(
                value="ethical everyone internet rection translation", label=True
            ),
            BinaryClassificationExample(
                value="ports changes president requisites eared", label=False
            ),
            BinaryClassificationExample(
                value="bad hers haar constitutional mental", label=False
            ),
            BinaryClassificationExample(
                value="lier intelligence mining devices flush", label=False
            ),
            BinaryClassificationExample(
                value="culture fascist mallow medical three", label=True
            ),
            BinaryClassificationExample(
                value="grades availability crazy contained flake", label=False
            ),
            BinaryClassificationExample(value="amic ln owned plays saw", label=True),
            BinaryClassificationExample(
                value="affected occupied ador gently filename", label=False
            ),
            BinaryClassificationExample(
                value="email legates minute umbo within", label=True
            ),
            BinaryClassificationExample(
                value="minute ping pointers toe wang", label=True
            ),
            BinaryClassificationExample(
                value="dimensional ghost haust picked same", label=True
            ),
            BinaryClassificationExample(
                value="reflect drawn players measures pants", label=False
            ),
            BinaryClassificationExample(
                value="nl virtual plex treatment mods", label=False
            ),
            BinaryClassificationExample(value="kit rd tp vier yond", label=True),
            BinaryClassificationExample(
                value="bons designed relative spring stress", label=True
            ),
            BinaryClassificationExample(
                value="nutrition ental production surface along", label=False
            ),
            BinaryClassificationExample(
                value="rained sort shr images oho", label=False
            ),
            BinaryClassificationExample(
                value="carbon guards impact wich windows", label=True
            ),
            BinaryClassificationExample(
                value="acting ailed albeit rals sweet", label=True
            ),
            BinaryClassificationExample(
                value="stairs modified aryl yes haired", label=False
            ),
            BinaryClassificationExample(
                value="reality ruly deck bike ream", label=False
            ),
            BinaryClassificationExample(
                value="bridge corn lake living prints", label=True
            ),
        ),
        chain_of_thought=(
            "The failing examples 'hend ville simple scroll roots' and 'nl virtual plex"
            " treatment mods' don't share any words. The passing examples 'culture"
            " fascist mallow medical three' and 'abilities forums nit utch wich' also"
            " don't share any words. This implies that the pattern isn't of the form"
            " 'contains the word W' or 'repeats the word W'. The passing example 'bons"
            " designed relative spring stress' and failing example 'nutrition ental"
            " production surface along' both contain 5 words. Therefore the pattern"
            " isn't of the form 'contains exactly X words'. The passing example 'email"
            " legates minute umbo within' starts with the word 'email' then contains"
            " the word 'legates' and ends with the word 'within'. These words are in"
            " alphabetical order. The failing example 'grades availability crazy"
            " contained flake' starts with the word 'grades' then contains the word"
            " 'availability' and ends with the word 'flake'. These words are not in"
            " alphabetical order. Therefore the pattern is that the words in the input"
            " must be in alphabetical order. Therefore the label for 'bridge corn lake"
            " living prints' is `True`."
        ),
    ),
    rules.contains_a_french_word: BinaryClassificationTask(
        instruction=(
            "The following inputs are labelled 'True' if they match a pattern and"
            " 'False' otherwise. The pattern is known to be very simple and explainable"
            " in plain English. Label the remaining inputs according to the pattern."
        ),
        examples=(
            BinaryClassificationExample(
                value="topic onder busters beans airs", label=True
            ),
            BinaryClassificationExample(
                value="split secondary basketball established financial", label=False
            ),
            BinaryClassificationExample(
                value="gio ais heavy wolves latest", label=False
            ),
            BinaryClassificationExample(
                value="birth odore their holding umps", label=True
            ),
            BinaryClassificationExample(value="hua your task oman wb", label=True),
            BinaryClassificationExample(
                value="rance cession conscious democratic wyn", label=True
            ),
            BinaryClassificationExample(
                value="etta altern poons warming stood", label=False
            ),
            BinaryClassificationExample(value="taker rix zona aryl apolog", label=True),
            BinaryClassificationExample(
                value="hent gradient aggressive usage lasting", label=False
            ),
            BinaryClassificationExample(value="laughter ln opal rn which", label=False),
            BinaryClassificationExample(
                value="features gage eely asia missions", label=True
            ),
            BinaryClassificationExample(
                value="covered sized venge odore iago", label=True
            ),
            BinaryClassificationExample(
                value="racial legates either hester styles", label=False
            ),
            BinaryClassificationExample(
                value="closure oser input iii potion", label=True
            ),
            BinaryClassificationExample(
                value="folios send paying etta connected", label=False
            ),
            BinaryClassificationExample(
                value="nothing sf reasonable ocre zed", label=True
            ),
            BinaryClassificationExample(
                value="amorph intention absolutely sis rama", label=True
            ),
            BinaryClassificationExample(
                value="ashes md training ocracy advertising", label=False
            ),
            BinaryClassificationExample(
                value="amins acca provided committee licks", label=False
            ),
            BinaryClassificationExample(value="ety warning guns lists tp", label=False),
            BinaryClassificationExample(
                value="arson rase toggle first ambling", label=True
            ),
            BinaryClassificationExample(
                value="opter program wt ket medium", label=True
            ),
            BinaryClassificationExample(value="rc avis coins dain jing", label=False),
            BinaryClassificationExample(
                value="provided owing pared frey layout", label=False
            ),
            BinaryClassificationExample(
                value="hooting rament oots alien ket", label=True
            ),
            BinaryClassificationExample(
                value="paste worn echo porter ocre", label=True
            ),
            BinaryClassificationExample(
                value="odium things marine izzard powder", label=True
            ),
            BinaryClassificationExample(
                value="annot orf thirds wolves someone", label=False
            ),
            BinaryClassificationExample(value="airy onder dale ibo ether", label=True),
            BinaryClassificationExample(
                value="bors bons etta defined agonal", label=False
            ),
            BinaryClassificationExample(
                value="default rums million bits protected", label=False
            ),
            BinaryClassificationExample(
                value="social casting basketball cars index", label=False
            ),
        ),
        chain_of_thought=(
            "The failing examples 'racial legates either hester styles' and 'provided"
            " owing pared frey layout' don't share any words. The passing examples"
            " 'opter program wt ket medium' and 'covered sized venge odore iago' also"
            " don't share any words. This implies that the pattern isn't of the form"
            " 'contains the word W'. However, the passing examples both contain French"
            " words: 'opter' and 'odore'. All passing examples contain French words and"
            " all failing examples don't contain French words. Therefore the pattern is"
            " that the input must contain a French word. Therefore the label for"
            " 'social casting basketball cars index' is `False`."
        ),
    ),
    rules.contains_the_word_w_within_x_positions_of_the_word_w_prime: BinaryClassificationTask(
        instruction=(
            "The following inputs are labelled 'True' if they match a pattern and"
            " 'False' otherwise. The pattern is known to be very simple and explainable"
            " in plain English. Label the remaining inputs according to the pattern."
        ),
        examples=(
            BinaryClassificationExample(
                value="oots fruit extremely scription graph", label=True
            ),
            BinaryClassificationExample(
                value="responsive extremely days scription bolt", label=False
            ),
            BinaryClassificationExample(
                value="extremely forth ilion casts scription", label=False
            ),
            BinaryClassificationExample(
                value="extremely oyer ctrl handled scription", label=False
            ),
            BinaryClassificationExample(
                value="extremely agency methyl scription minimum", label=False
            ),
            BinaryClassificationExample(
                value="young ahu patient extremely scription", label=True
            ),
            BinaryClassificationExample(
                value="pattern extremely placed conscious scription", label=False
            ),
            BinaryClassificationExample(
                value="such filled extremely dead scription", label=False
            ),
            BinaryClassificationExample(
                value="hod democracy roach extremely scription", label=True
            ),
            BinaryClassificationExample(
                value="spread cham extremely rill scription", label=False
            ),
            BinaryClassificationExample(
                value="rake extremely cedes scription outer", label=False
            ),
            BinaryClassificationExample(
                value="extremely character alist laughs scription", label=False
            ),
            BinaryClassificationExample(
                value="leaders roups extremely scription hler", label=True
            ),
            BinaryClassificationExample(
                value="animal unknown inline extremely scription", label=True
            ),
            BinaryClassificationExample(
                value="extremely sand weed bottom scription", label=False
            ),
            BinaryClassificationExample(
                value="animous happy also extremely scription", label=True
            ),
            BinaryClassificationExample(
                value="meal extremely laws scription execute", label=False
            ),
            BinaryClassificationExample(
                value="ply fs status extremely scription", label=True
            ),
            BinaryClassificationExample(
                value="extremely scription based crew trade", label=True
            ),
            BinaryClassificationExample(
                value="extremely approximately abies scription armed", label=False
            ),
            BinaryClassificationExample(
                value="stated extremely scription rived cation", label=True
            ),
            BinaryClassificationExample(
                value="marine query runs extremely scription", label=True
            ),
            BinaryClassificationExample(
                value="amorph extremely respond ported scription", label=False
            ),
            BinaryClassificationExample(
                value="please aches among extremely scription", label=True
            ),
            BinaryClassificationExample(
                value="extremely scription faced frey pillar", label=True
            ),
            BinaryClassificationExample(
                value="hillary security derived extremely scription", label=True
            ),
            BinaryClassificationExample(
                value="vector safety extremely painted scription", label=False
            ),
            BinaryClassificationExample(
                value="boarding ammu choice extremely scription", label=True
            ),
            BinaryClassificationExample(
                value="mud outing acca extremely scription", label=True
            ),
            BinaryClassificationExample(
                value="sym lasting extremely olson scription", label=False
            ),
            BinaryClassificationExample(
                value="commerce members grid extremely scription", label=True
            ),
            BinaryClassificationExample(
                value="extremely features goal capacity scription", label=False
            ),
        ),
        chain_of_thought=(
            "The passing examples 'oots fruit extremely scription graph' and 'young ahu"
            " patient extremely scription' both contain the words 'extremely' and"
            " 'scription'. The failing examples 'extremely agency methyl scription"
            " minimum' and 'spread cham extremely rill scription' both contain the"
            " words 'extremely' and 'scription'. This implies that the pattern can't be"
            " of the form 'contains the words W and W''. However, it might be a"
            " variation of this pattern. In the passing examples, the words 'extremely'"
            " and 'scription' are neighbors. In the failing examples, the words"
            " 'extremely' and 'scription' are not neighbors. Therefore the pattern is"
            " that the words W and W' must be neighbors. Therefore the label for"
            " 'extremely features goal capacity scription' is `False`."
        ),
    ),
    rules.repeats_the_word_w_and_contains_a_french_word: BinaryClassificationTask(
        instruction=(
            "The following inputs are labelled 'True' if they match a pattern and"
            " 'False' otherwise. The pattern is known to be very simple and explainable"
            " in plain English. Label the remaining inputs according to the pattern."
        ),
        examples=(
            BinaryClassificationExample(
                value="ging odore minute monitor jack", label=False
            ),
            BinaryClassificationExample(
                value="cube domain marine intuitive dry", label=False
            ),
            BinaryClassificationExample(
                value="existent aten same camera assert", label=False
            ),
            BinaryClassificationExample(
                value="ramer humane existent existent existent", label=True
            ),
            BinaryClassificationExample(
                value="cow existent court during bishop", label=False
            ),
            BinaryClassificationExample(
                value="integer existent hua irk pole", label=False
            ),
            BinaryClassificationExample(
                value="missions acre existent existent oit", label=True
            ),
            BinaryClassificationExample(
                value="existent existent existent existent existent", label=False
            ),
            BinaryClassificationExample(
                value="country existent rase rising existent", label=True
            ),
            BinaryClassificationExample(
                value="existent invest balanced existent semble", label=True
            ),
            BinaryClassificationExample(
                value="shell adding existent existent usions", label=True
            ),
            BinaryClassificationExample(
                value="existent amins nie hp existent", label=True
            ),
            BinaryClassificationExample(
                value="context existent important existent hazard", label=False
            ),
            BinaryClassificationExample(
                value="poll pointers existent odore existent", label=True
            ),
            BinaryClassificationExample(
                value="levant violent forcer stall existent", label=False
            ),
            BinaryClassificationExample(
                value="natal existent official existent existent", label=False
            ),
            BinaryClassificationExample(
                value="existent gaming existent existent venge", label=True
            ),
            BinaryClassificationExample(
                value="existent existent iris blocking skinned", label=False
            ),
            BinaryClassificationExample(
                value="things existent mph existent hua", label=True
            ),
            BinaryClassificationExample(
                value="existent lier wreck clamation methyl", label=False
            ),
            BinaryClassificationExample(
                value="demon existent existent existent rance", label=True
            ),
            BinaryClassificationExample(
                value="pared existent rade existent ratch", label=True
            ),
            BinaryClassificationExample(
                value="urger existent rals zone forth", label=False
            ),
            BinaryClassificationExample(
                value="existent combe dominated levant rants", label=False
            ),
            BinaryClassificationExample(
                value="existent existent body degree rament", label=True
            ),
            BinaryClassificationExample(
                value="deep existent completely journal von", label=False
            ),
            BinaryClassificationExample(
                value="existent marine existent night graph", label=True
            ),
            BinaryClassificationExample(
                value="existent fare existent existent oser", label=True
            ),
            BinaryClassificationExample(
                value="existent urger maxwell eligible existent", label=True
            ),
            BinaryClassificationExample(
                value="existent existent existent painted rade", label=True
            ),
            BinaryClassificationExample(
                value="ails evidence winter mund existent", label=False
            ),
            BinaryClassificationExample(
                value="existent oit nesses dain ln", label=False
            ),
        ),
        chain_of_thought=(
            "The passing examples 'existent marine existent night graph' and 'existent"
            " gaming existent existent venge' both repeat the word 'existent'. The"
            " failing examples 'existent lier wreck clamation methyl' and 'urger"
            " existent rals zone forth' both contain the word 'existent' but don't"
            " repeat it. This implies that the pattern is of the form 'repeats the word"
            " W'. However, the example 'existent existent existent existent existent'"
            " is a failing example. This implies that the pattern isn't of the form"
            " 'repeats the word W' but a variation of this pattern. The passing"
            " examples also contain the French words 'marine' and 'vange'. All passing"
            " examples repeat the word 'existent' and contain French words and all"
            " failing examples don't repeat the word 'existent' or contain French"
            " words. Therefore the pattern is 'repeats the word W and contains a French"
            " word'. Therefore the label for 'existent oit nesses dain ln' is `False`."
        ),
    ),
}

###############################
# FREEFORM ARTICULATION TASKS #
###############################

HANDWRITTEN_FREEFORM_ARTICULATION_TASKS_WITH_CHAIN_OF_THOUGHT = {
    rules.contains_the_word_w_and_a_digit: FreeformExplanationTask(
        instruction=(
            "The following inputs are labelled 'True' if they match a pattern and"
            " 'False' otherwise. The pattern is known to be very simple and explainable"
            " in plain English. Label the remaining inputs according to the pattern."
        ),
        classification_task=BinaryClassificationTask(
            examples=(
                BinaryClassificationExample(
                    value="rub happy give laugh city", label=False
                ),
                BinaryClassificationExample(
                    value="depend century space fish section", label=False
                ),
                BinaryClassificationExample(
                    value="several voice learn 9 section", label=True
                ),
                BinaryClassificationExample(
                    value="5 apple finger hour moon", label=False
                ),
                BinaryClassificationExample(
                    value="neck 7 chance chair section", label=True
                ),
                BinaryClassificationExample(
                    value="section first floor carry sell", label=False
                ),
                BinaryClassificationExample(
                    value="quart section made vary 2", label=True
                ),
                BinaryClassificationExample(
                    value="leave cloud summer 9 section", label=True
                ),
                BinaryClassificationExample(
                    value="learn food magnet arrive size", label=False
                ),
                BinaryClassificationExample(
                    value="clean put noon 8 section", label=True
                ),
                BinaryClassificationExample(
                    value="history phrase practice section 6", label=True
                ),
                BinaryClassificationExample(
                    value="danger unit chance 3 section", label=True
                ),
                BinaryClassificationExample(
                    value="baby summer print shall 0", label=False
                ),
                BinaryClassificationExample(
                    value="bank section dog school wall", label=False
                ),
                BinaryClassificationExample(
                    value="gave add quiet 8 section", label=True
                ),
                BinaryClassificationExample(
                    value="gun children noise section six", label=False
                ),
            ),
            instruction=None,
            chain_of_thought=None,
        ),
        explanation="The input contains the word `section` and a digit",
        chain_of_thought=(
            "All passing examples contain the word `section`. Some failing examples"
            " also contain the word `section`. Therefore the pattern can't simply"
            " depend on the presence or absence of the word `section`. All passing"
            " examples contain a number. Some failing examples contain the word"
            " `section` or a number, but not both. Therefore the rule can't be"
            ' "contains the word `section` or a number". The rule is therefore'
            ' "contains the word `month` or a number".'
        ),
    ),
    rules.words_are_in_alphabetical_order: FreeformExplanationTask(
        instruction=(
            "The following inputs are labelled 'True' if they match a pattern and"
            " 'False' otherwise. The pattern is known to be very simple and explainable"
            " in plain English. Label the remaining inputs according to the pattern."
        ),
        classification_task=BinaryClassificationTask(
            examples=(
                BinaryClassificationExample(
                    value="about direct found heard weather", label=True
                ),
                BinaryClassificationExample(
                    value="were object mix require life", label=False
                ),
                BinaryClassificationExample(
                    value="little consider thus stand dictionary", label=False
                ),
                BinaryClassificationExample(
                    value="consider down roll told unit", label=True
                ),
                BinaryClassificationExample(
                    value="cotton enough find original shout", label=True
                ),
                BinaryClassificationExample(
                    value="hot wire allow prove back", label=False
                ),
                BinaryClassificationExample(
                    value="boat branch bright decimal job", label=True
                ),
                BinaryClassificationExample(
                    value="block cloud dollar fun safe", label=True
                ),
                BinaryClassificationExample(
                    value="settle rich felt noon real", label=False
                ),
                BinaryClassificationExample(
                    value="cloud effect often store value", label=True
                ),
                BinaryClassificationExample(
                    value="care fun gun silent those", label=True
                ),
                BinaryClassificationExample(
                    value="solution work fresh clock minute", label=False
                ),
                BinaryClassificationExample(
                    value="animal study light bring noun", label=False
                ),
                BinaryClassificationExample(
                    value="talk crease grew symbol against", label=False
                ),
                BinaryClassificationExample(
                    value="center game less meant shine", label=True
                ),
                BinaryClassificationExample(
                    value="travel order compare area roll", label=False
                ),
            ),
            instruction=None,
            chain_of_thought=None,
        ),
        explanation="The words in the input are in alphabetical order",
        chain_of_thought=(
            "Many passing examples don't have words in common. Many failing examples"
            " don't have words in common either. Therefore the pattern can't depend on"
            " the presence or absence of a word. The passing examples are in"
            " alphabetical order. The failing examples are not in alphabetical order."
            " Therefore the pattern is that the examples are in alphabetical order."
        ),
    ),
    rules.contains_the_word_w_followed_by_w_prime: FreeformExplanationTask(
        instruction=(
            "The following inputs are labelled 'True' if they match a pattern and"
            " 'False' otherwise. The pattern is known to be very simple and explainable"
            " in plain English. Label the remaining inputs according to the pattern."
        ),
        classification_task=BinaryClassificationTask(
            examples=(
                BinaryClassificationExample(
                    value="west colony key sugar nation", label=False
                ),
                BinaryClassificationExample(
                    value="stop bell subject chance decimal", label=True
                ),
                BinaryClassificationExample(
                    value="chance divide weight bell push", label=False
                ),
                BinaryClassificationExample(
                    value="but truck bell chance post", label=True
                ),
                BinaryClassificationExample(
                    value="indicate carry swim run shore", label=False
                ),
                BinaryClassificationExample(
                    value="bell clothe control trouble chance", label=True
                ),
                BinaryClassificationExample(
                    value="toward area chance subtract bell", label=False
                ),
                BinaryClassificationExample(
                    value="bell prove wire step chance", label=True
                ),
                BinaryClassificationExample(
                    value="chance still bell between poor", label=False
                ),
                BinaryClassificationExample(
                    value="bell chance help total broad", label=True
                ),
                BinaryClassificationExample(
                    value="bell feed leave chance real", label=True
                ),
                BinaryClassificationExample(
                    value="bell silent second chance block", label=True
                ),
                BinaryClassificationExample(
                    value="radio bell chance ask want", label=True
                ),
                BinaryClassificationExample(
                    value="same locate strange determine indicate", label=False
                ),
                BinaryClassificationExample(
                    value="mix chance bit bell stick", label=False
                ),
                BinaryClassificationExample(
                    value="charge chance bell lone major", label=False
                ),
            ),
            instruction=None,
            chain_of_thought=None,
        ),
        explanation="The input contains the word `bell` before the word `chance`",
        chain_of_thought=(
            "All passing examples contain the words `bell` and `chance`. Many failing"
            " examples also contain the words `bell` and `chance`. Therefore the"
            " pattern can't simply depend on the presence or absence of these two"
            " words. The passing examples contain the words `bell` and `chance` in that"
            " order. The failing examples contain the words `bell` and `chance` in a"
            " different order, or not at all. Therefore the pattern is that the"
            " examples contain the words `bell` and `chance` in that order."
        ),
    ),
    rules.contains_a_french_word: FreeformExplanationTask(
        instruction=(
            "The following inputs are labelled 'True' if they match a pattern and"
            " 'False' otherwise. The pattern is known to be very simple and explainable"
            " in plain English. Label the remaining inputs according to the pattern."
        ),
        classification_task=BinaryClassificationTask(
            examples=(
                BinaryClassificationExample(
                    value="spot oit afraid fit tail", label=True
                ),
                BinaryClassificationExample(
                    value="melody division come experiment about", label=False
                ),
                BinaryClassificationExample(
                    value="five raits chart fast deep", label=True
                ),
                BinaryClassificationExample(
                    value="fine ramer people both held", label=True
                ),
                BinaryClassificationExample(
                    value="zona chick king wire gather", label=True
                ),
                BinaryClassificationExample(
                    value="hunt happen often order last", label=False
                ),
                BinaryClassificationExample(
                    value="question hope grow store possible", label=False
                ),
                BinaryClassificationExample(
                    value="family heat circle suggest clothe", label=False
                ),
                BinaryClassificationExample(
                    value="same milk edge column woman", label=False
                ),
                BinaryClassificationExample(
                    value="property object best fresh beat", label=False
                ),
                BinaryClassificationExample(
                    value="match type earth forcer decimal", label=True
                ),
                BinaryClassificationExample(
                    value="forest crease gage with teeth", label=True
                ),
                BinaryClassificationExample(
                    value="urger nie degree hua believe", label=True
                ),
                BinaryClassificationExample(
                    value="wife power your evening steel", label=False
                ),
                BinaryClassificationExample(
                    value="cow pair language hua cloud", label=True
                ),
                BinaryClassificationExample(
                    value="why certain bit weight shall", label=False
                ),
            ),
            instruction=None,
            chain_of_thought=None,
        ),
        explanation="The input contains a French word",
        chain_of_thought=(
            "No one word appears in inputs which are labelled as `True`. Likewise, no"
            " one word appears in inputs which are labelled as `False`. Therefore the"
            " rule can't be one which depends on the presence or absence of a word."
            ' Some inputs labelled `True` contain the words "gage", "ramer" and'
            ' "raits", which are all French words. All inputs labelled `False` don\'t'
            ' contain French words. Therefore the pattern is "contains a French word".'
        ),
    ),
    rules.word_w_appears_in_position_i: FreeformExplanationTask(
        instruction=(
            "The following inputs are labelled 'True' if they match a pattern and"
            " 'False' otherwise. The pattern is known to be very simple and explainable"
            " in plain English. Label the remaining inputs according to the pattern."
        ),
        classification_task=BinaryClassificationTask(
            examples=(
                BinaryClassificationExample(
                    value="king print correct thousand boy", label=True
                ),
                BinaryClassificationExample(
                    value="reason gave listen wrong science", label=False
                ),
                BinaryClassificationExample(
                    value="cell hand famous correct written", label=False
                ),
                BinaryClassificationExample(
                    value="carry blow correct wild point", label=True
                ),
                BinaryClassificationExample(
                    value="build wash bad position live", label=False
                ),
                BinaryClassificationExample(
                    value="soldier must correct property paint", label=True
                ),
                BinaryClassificationExample(
                    value="control few piece block correct", label=False
                ),
                BinaryClassificationExample(
                    value="roll found correct problem join", label=True
                ),
                BinaryClassificationExample(
                    value="correct determine map consider share", label=False
                ),
                BinaryClassificationExample(
                    value="double between correct letter spend", label=True
                ),
                BinaryClassificationExample(
                    value="correct green receive baby seven", label=False
                ),
                BinaryClassificationExample(
                    value="discuss bed correct radio them", label=True
                ),
                BinaryClassificationExample(
                    value="noun word correct string evening", label=True
                ),
                BinaryClassificationExample(
                    value="bed note which take sentence", label=False
                ),
                BinaryClassificationExample(
                    value="else push correct seat choose", label=True
                ),
                BinaryClassificationExample(
                    value="shore crease held that division", label=False
                ),
            ),
            instruction=None,
            chain_of_thought=None,
        ),
        explanation="The word `correct` is in position 2",
        chain_of_thought=(
            "The word `correct` appears in all passing and failing inputs. Therefore"
            " the pattern can't be one which depends solely of the presence of this"
            " word. In the passing inputs, `correct` appears as the third word. In the"
            " failing inputs, `correct` either doesn't appear or appears but never as"
            ' the third word. Therefore the pattern is "contains the word `correct` as'
            ' the third word".'
        ),
    ),
}
