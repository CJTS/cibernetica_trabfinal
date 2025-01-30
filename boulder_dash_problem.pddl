(define (problem boulder_dash_problem)
  (:domain boulder_dash)

  ;; Objects
  (:objects
    p1 - player
    d1 d2 - diamond
    r1 r2 - rock
    c1 c2 c3 c4 c5 c6 c7 c8 c9 - cell
  )

  ;; Initial State
  (:init
    (at p1 c1)
    (faced-right p1)
    ; (diamond-at d1 c3)
    ; (diamond-at d2 c4)
    ; (rock-at r1 c5)
    ; (rock-at r2 c6)
    ; (empty c2)
    (diamond-at d1 c6)
    (rock-at r1 c3)
    (empty c2)
    (empty c4)
    (empty c5)
    (empty c7)
    (empty c8)
    (empty c9)
    (adjacent-right c1 c2) (adjacent-right c2 c3) (adjacent-right c3 c4) (adjacent-right c4 c5) (adjacent-right c5 c6)
    (adjacent-left c2 c1) (adjacent-left c3 c2) (adjacent-left c4 c3) (adjacent-left c5 c4) (adjacent-left c6 c5)
    (adjacent-right c8 c9) (adjacent-down c2 c8) (adjacent-down c3 c9) (adjacent-down c7 c3)
    (adjacent-left c9 c8) (adjacent-up c8 c2) (adjacent-up c9 c3) (adjacent-up c3 c7)
  )

  ;; Goal State
  (:goal
    (and
      (collected d1)
    )
  )
)
