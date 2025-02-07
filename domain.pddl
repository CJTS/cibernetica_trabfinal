(define (domain boulder_dash)
    (:requirements :strips :typing :existential-preconditions :conditional-effects)
    (:types
        locatable - object
        player - locatable
        gem - locatable
        boulder - locatable
        dirt - locatable
        cell - place
    )

    (:predicates
        (at ?l - locatable ?c - cell)
        (connected-up ?c1 ?c2 - cell)
        (connected-down ?c1 ?c2 - cell)
        (connected-left ?c1 ?c2 - cell)
        (connected-right ?c1 ?c2 - cell)
        (oriented-up ?p - player)
        (oriented-down ?p - player)
        (oriented-left ?p - player)
        (oriented-right ?p - player)
        (terrain-wall ?c - cell)
        (terrain-empty ?c - cell)
        (got ?d - gem)
    )

    (:action move-up
        :parameters (?p - player ?c1 ?c2 - cell ?gem - gem)
        :precondition (and
            (at ?p ?c1)
            (connected-up ?c1 ?c2)
            (not (exists
                    (?b - boulder)
                    (at ?b ?c2)
                )
            )
            (not (terrain-wall ?c2))
        )
        :effect (and
            (when ;oriented correctly and has gem
                (and
                    (oriented-up ?p)
                    (at ?gem ?c2)
                )
                (and
                    (not (at ?p ?c1))
                    (not (at ?gem ?c2))
                    (at ?p ?c2)
                    (got ?gem)
                )
            )
            (when ;oriented correctly and has no gem
                (and
                    (oriented-up ?p)
                    (not (at ?gem ?c2))
                )
                (and
                    (not (at ?p ?c1))
                    (at ?p ?c2)
                )
            )
            (when ;not oriented correctly
                (not (oriented-up ?p))
                (and
                    (oriented-up ?p)
                    (not (oriented-down ?p))
                    (not (oriented-left ?p))
                    (not (oriented-right ?p))
                )
            )
        )
    )

    (:action move-down
        :parameters (?p - player ?c1 ?c2 - cell ?gem - gem)
        :precondition (and
            (at ?p ?c1)
            (connected-down ?c1 ?c2)
            (not (exists
                    (?b - boulder)
                    (at ?b ?c2)
                )
            )
            (not (terrain-wall ?c2))
        )
        :effect (and
            (when ;oriented correctly and has gem
                (and
                    (oriented-down ?p)
                    (at ?gem ?c2)
                )
                (and
                    (not (at ?p ?c1))
                    (not (at ?gem ?c2))
                    (at ?p ?c2)
                    (got ?gem)
                )
            )
            (when ;oriented correctly and has no gem
                (and
                    (oriented-down ?p)
                    (not (at ?gem ?c2))
                )
                (and
                    (not (at ?p ?c1))
                    (at ?p ?c2)
                )
            )
            (when ;not oriented correctly
                (not (oriented-down ?p))
                (and
                    (oriented-down ?p)
                    (not (oriented-up ?p))
                    (not (oriented-left ?p))
                    (not (oriented-right ?p))
                )
            )
        )
    )

    (:action move-left
        :parameters (?p - player ?c1 ?c2 - cell ?gem - gem)
        :precondition (and
            (at ?p ?c1)
            (connected-left ?c1 ?c2)
            (not (exists
                    (?b - boulder)
                    (at ?b ?c2)
                )
            )
            (not (terrain-wall ?c2))
        )
        :effect (and
            (when ;oriented correctly and has gem
                (and
                    (oriented-left ?p)
                    (at ?gem ?c2)
                )
                (and
                    (not (at ?p ?c1))
                    (not (at ?gem ?c2))
                    (at ?p ?c2)
                    (got ?gem)
                )
            )
            (when ;oriented correctly and has no gem
                (and
                    (oriented-left ?p)
                    (not (at ?gem ?c2))
                )
                (and
                    (not (at ?p ?c1))
                    (at ?p ?c2)
                )
            )
            (when ;not oriented correctly
                (not (oriented-left ?p))
                (and
                    (oriented-left ?p)
                    (not (oriented-down ?p))
                    (not (oriented-up ?p))
                    (not (oriented-right ?p))
                )
            )
        )
    )

    (:action move-right
        :parameters (?p - player ?c1 ?c2 - cell ?gem - gem)
        :precondition (and
            (at ?p ?c1)
            (connected-right ?c1 ?c2)
            (not (exists
                    (?b - boulder)
                    (at ?b ?c2)
                )
            )
            (not (terrain-wall ?c2))
        )
        :effect (and
            (when ;oriented correctly and has gem
                (and
                    (oriented-right ?p)
                    (at ?gem ?c2)
                )
                (and
                    (not (at ?p ?c1))
                    (not (at ?gem ?c2))
                    (at ?p ?c2)
                    (got ?gem)
                )
            )
            (when ;oriented correctly and has no gem
                (and
                    (oriented-right ?p)
                    (not (at ?gem ?c2))
                )
                (and
                    (not (at ?p ?c1))
                    (at ?p ?c2)
                )
            )
            (when ;not oriented correctly
                (not (oriented-right ?p))
                (and
                    (oriented-right ?p)
                    (not (oriented-down ?p))
                    (not (oriented-left ?p))
                    (not (oriented-up ?p))
                )
            )
        )
    )

    (:action use-up
        :parameters (?p - player ?b - boulder ?c1 ?c2 - cell)
        :precondition (and
            (at ?p ?c1)
            (at ?b ?c2)
            (oriented-up ?p)
            (connected-up ?c1 ?c2)
        )
        :effect (and
            (not (at ?b ?c2))
        )
    )
    (:action use-down
        :parameters (?p - player ?b - boulder ?c1 ?c2 - cell)
        :precondition (and
            (at ?p ?c1)
            (at ?b ?c2)
            (oriented-down ?p)
            (connected-down ?c1 ?c2)
        )
        :effect (and
            (not (at ?b ?c2))
        )
    )
    (:action use-left
        :parameters (?p - player ?b - boulder ?c1 ?c2 - cell)
        :precondition (and
            (at ?p ?c1)
            (at ?b ?c2)
            (oriented-left ?p)
            (connected-left ?c1 ?c2)
        )
        :effect (and
            (not (at ?b ?c2))
        )
    )
    (:action use-right
        :parameters (?p - player ?b - boulder ?c1 ?c2 - cell)
        :precondition (and
            (at ?p ?c1)
            (at ?b ?c2)
            (oriented-right ?p)
            (connected-right ?c1 ?c2)
        )
        :effect (and
            (not (at ?b ?c2))
        )
    )
)