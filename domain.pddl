(define (domain boulder_dash)
    (:requirements :strips :typing)
    (:types
        player diamond rock cell
    )

    (:predicates
        (at ?l − Locatable ?c − Cell)
        (connected−up ?c1 ?c2 − Cell)
        (connected−down ?c1 ?c2 − Cell)
        (connected−left ?c1 ?c2 − Cell)
        (connected−right ?c1 ?c2 − Cell)
        (oriented−up ?p − Player)
        (oriented−down ?p − Player)
        (oriented−left ?p − Player)
        (oriented−right ?p − Player)
    )

    (:action move−up
        :parameters (?p − Player ?c1 ?c2 − Cell)
        :precondition (and
            (at ?p ?c1)
            (oriented−up ?p)
            (connected−up ?c1 ?c2)
            (not (exists
                    (?b − Boulder)
                    (at ?b ?c2)
                )
            )
            (not (terrain−wall ?c2))
        )
        :effect (and
            (when
                (not ( terrain−empty ?c2))
                (terrain−empty ?c2)
            )
            (not (at ?p ?c1))
            (at ?p ?c2)
        )
    )

    (:action move−down
        :parameters (?p − Player ?c1 ?c2 − Cell)
        :precondition (and
            (at ?p ?c1)
            (oriented−down ?p)
            (connected−down ?c1 ?c2)
            (not (exists
                    (?b − Boulder)
                    (at ?b ?c2)
                )
            )
            (not (terrain−wall ?c2))
        )
        :effect (and
            (when
                (not ( terrain−empty ?c2))
                (terrain−empty ?c2)
            )
            (not (at ?p ?c1))
            (at ?p ?c2)
        )
    )

    (:action move−left
        :parameters (?p − Player ?c1 ?c2 − Cell)
        :precondition (and
            (at ?p ?c1)
            (oriented−left ?p)
            (connected−left ?c1 ?c2)
            (not (exists
                    (?b − Boulder)
                    (at ?b ?c2)
                )
            )
            (not (terrain−wall ?c2))
        )
        :effect (and
            (when
                (not ( terrain−empty ?c2))
                (terrain−empty ?c2)
            )
            (not (at ?p ?c1))
            (at ?p ?c2)
        )
    )

    (:action move−right
        :parameters (?p − Player ?c1 ?c2 − Cell)
        :precondition (and
            (at ?p ?c1)
            (oriented−right ?p)
            (connected−right ?c1 ?c2)
            (not (exists
                    (?b − Boulder)
                    (at ?b ?c2)
                )
            )
            (not (terrain−wall ?c2))
        )
        :effect (and
            (when
                (not ( terrain−empty ?c2))
                (terrain−empty ?c2)
            )
            (not (at ?p ?c1))
            (at ?p ?c2)
        )
    )

    (:action use
        :parameters (?p − Player ?b − boulder ?c1 ?c2 − Cell)
        :precondition (and
            (at ?p ?c1)
            (at ?b ?c2)
            (oriented−up ?p)
            (connected−up ?c1 ?c2)
        )
        :effect (and
            (not (at ?b ?c2))
        )
    )
)