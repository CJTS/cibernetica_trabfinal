(define (domain boulder_dash)
  (:requirements :strips :typing)
  (:types 
    locatable - object
    player gem rock dirt - locatable
    cell - place
  )
  ; (:constants up down left right - orientation)

  ;; Predicates
  (:predicates
    (at ?l - locatable ?c - cell)            ;; Player is at a cell
    (empty ?c - cell)                     ;; A cell is empty
    (adjacent-up ?c1 - cell ?c2 - cell)   ;; Two cells are adjacent
    (adjacent-down ?c1 - cell ?c2 - cell)   ;; Two cells are adjacent
    (adjacent-left ?c1 - cell ?c2 - cell)   ;; Two cells are adjacent
    (adjacent-right ?c1 - cell ?c2 - cell)   ;; Two cells are adjacent
    (collected ?d - gem)              ;; Diamond has been collected
    (faced-up ?p - player)
    (faced-down ?p - player)
    (faced-right ?p - player)
    (faced-left ?p - player)
  )

  ;; Actions
  (:action face-up
    :parameters (?p - player)
    :precondition (and (not (faced-up ?p)))
    :effect (and
      (faced-up ?p)
      (not (faced-down ?p))
      (not (faced-left ?p))
      (not (faced-right ?p))
    )
  )
  (:action face-right
    :parameters (?p - player)
    :precondition (and (not (faced-right ?p)))
    :effect (and
      (faced-right ?p)
      (not (faced-down ?p))
      (not (faced-left ?p))
      (not (faced-up ?p))
    )
  )
  (:action face-down
    :parameters (?p - player)
    :precondition (and (not (faced-down ?p)))
    :effect (and
      (faced-down ?p)
      (not (faced-up ?p))
      (not (faced-left ?p))
      (not (faced-right ?p))
    )
  )
  (:action face-left
    :parameters (?p - player)
    :precondition (and (not (faced-left ?p)))
    :effect (and
      (faced-left ?p)
      (not (faced-down ?p))
      (not (faced-up ?p))
      (not (faced-right ?p))
    )
  )

  ;; Actions
  (:action move-up
    :parameters (?p - player ?from - cell ?to - cell)
    :precondition (and (faced-up ?p) (at ?p ?from) (empty ?to) (adjacent-up ?from ?to))
    :effect (and (not (at ?p ?from)) (at ?p ?to) (empty ?from) (not (empty ?to)))
  )
  (:action move-down
    :parameters (?p - player ?from - cell ?to - cell)
    :precondition (and (faced-down ?p) (at ?p ?from) (empty ?to) (adjacent-down ?from ?to))
    :effect (and (not (at ?p ?from)) (at ?p ?to) (empty ?from) (not (empty ?to)))
  )
  (:action move-left
    :parameters (?p - player ?from - cell ?to - cell)
    :precondition (and (faced-left ?p) (at ?p ?from) (empty ?to) (adjacent-left ?from ?to))
    :effect (and (not (at ?p ?from)) (at ?p ?to) (empty ?from) (not (empty ?to)))
  )
  (:action move-right
    :parameters (?p - player ?from - cell ?to - cell)
    :precondition (and (faced-right ?p) (at ?p ?from) (empty ?to) (adjacent-right ?from ?to))
    :effect (and (not (at ?p ?from)) (at ?p ?to) (empty ?from) (not (empty ?to)))
  )

  (:action dig-up
    :parameters (?p - player ?from - cell ?to - cell)
    :precondition (and (faced-up ?p) (at ?p ?from) (dirt-at ?to) (adjacent-up ?from ?to))
    :effect (and (not (at ?p ?from)) (at ?p ?to) (empty ?from) (not (dirt-at ?to)))
  )
  (:action dig-down
    :parameters (?p - player ?from - cell ?to - cell)
    :precondition (and (faced-down ?p) (at ?p ?from) (dirt-at ?to) (adjacent-down ?from ?to))
    :effect (and (not (at ?p ?from)) (at ?p ?to) (empty ?from) (not (dirt-at ?to)))
  )
  (:action dig-left
    :parameters (?p - player ?from - cell ?to - cell)
    :precondition (and (faced-left ?p) (at ?p ?from) (dirt-at ?to) (adjacent-left ?from ?to))
    :effect (and (not (at ?p ?from)) (at ?p ?to) (empty ?from) (not (dirt-at ?to)))
  )
  (:action dig-right
    :parameters (?p - player ?from - cell ?to - cell)
    :precondition (and (faced-right ?p) (at ?p ?from) (dirt-at ?to) (adjacent-right ?from ?to))
    :effect (and (not (at ?p ?from)) (at ?p ?to) (empty ?from) (not (dirt-at ?to)))
  )

  (:action collect-gem-up
    :parameters (?p - player ?from - cell ?to - cell ?d - gem)
    :precondition (and
      (at ?p ?from)
      (gem-at ?d ?to)
      (adjacent-up ?from ?to)
      (faced-up ?p)
    )
    :effect (and (collected ?d) (not (gem-at ?d ?to)) (empty ?to) (not (at ?p ?from)) (at ?p ?to))
  )
  (:action collect-gem-down
    :parameters (?p - player ?from - cell ?to - cell ?d - gem)
    :precondition (and
      (at ?p ?from)
      (gem-at ?d ?to)
      (adjacent-down ?from ?to)
      (faced-down ?p)
    )
    :effect (and (collected ?d) (not (gem-at ?d ?to)) (empty ?to) (not (at ?p ?from)) (at ?p ?to))
  )
  (:action collect-gem-left
    :parameters (?p - player ?from - cell ?to - cell ?d - gem)
    :precondition (and
      (at ?p ?from)
      (gem-at ?d ?to)
      (adjacent-left ?from ?to)
      (faced-left ?p)
    )
    :effect (and (collected ?d) (not (gem-at ?d ?to)) (empty ?to) (not (at ?p ?from)) (at ?p ?to))
  )
  (:action collect-gem-right
    :parameters (?p - player ?from - cell ?to - cell ?d - gem)
    :precondition (and
      (at ?p ?from)
      (gem-at ?d ?to)
      (adjacent-right ?from ?to)
      (faced-right ?p)
    )
    :effect (and (collected ?d) (not (gem-at ?d ?to)) (empty ?to) (not (at ?p ?from)) (at ?p ?to))
  )

  ; (:action push-rock-up
  ;   :parameters (?p - player ?playerCell - cell ?rockCell - cell ?pushedCell - cell ?rock - rock)
  ;   :precondition (and
  ;     (at ?p ?playerCell)
  ;     (rock-at ?rock ?rockCell)
  ;     (empty ?pushedCell)
  ;     (faced-up ?p)
  ;     (adjacent-up ?playerCell ?rockCell)
  ;     (adjacent-up ?rockCell ?pushedCell)
  ;   )
  ;   :effect (and
  ;     (not (at ?p ?playerCell))
  ;     (at ?p ?rockCell)
  ;     (not (rock-at ?rock ?rockCell))
  ;     (rock-at ?rock ?pushedCell)
  ;     (not (empty ?pushedCell))
  ;     (empty ?playerCell)
  ;   )
  ; )

  (:action push-rock-right
    :parameters (?p - player ?playerCell - cell ?rockCell - cell ?pushedCell - cell ?rock - rock)
    :precondition (and
      (at ?p ?playerCell)
      (rock-at ?rock ?rockCell)
      (empty ?pushedCell)
      (faced-right ?p)
      (adjacent-right ?playerCell ?rockCell)
      (adjacent-right ?rockCell ?pushedCell)
    )
    :effect (and
      (not (at ?p ?playerCell))
      (at ?p ?rockCell)
      (not (rock-at ?rock ?rockCell))
      (rock-at ?rock ?pushedCell)
      (not (empty ?pushedCell))
      (empty ?playerCell)
    )
  )

  ; (:action push-rock-down
  ;   :parameters (?p - player ?playerCell - cell ?rockCell - cell ?pushedCell - cell ?rock - rock)
  ;   :precondition (and
  ;     (at ?p ?playerCell)
  ;     (rock-at ?rock ?rockCell)
  ;     (empty ?pushedCell)
  ;     (faced-down ?p)
  ;     (adjacent-down ?playerCell ?rockCell)
  ;     (adjacent-down ?rockCell ?pushedCell)
  ;   )
  ;   :effect (and
  ;     (not (at ?p ?playerCell))
  ;     (at ?p ?rockCell)
  ;     (not (rock-at ?rock ?rockCell))
  ;     (rock-at ?rock ?pushedCell)
  ;     (not (empty ?pushedCell))
  ;     (empty ?playerCell)
  ;   )
  ; )

  (:action push-rock-left
    :parameters (?p - player ?playerCell - cell ?rockCell - cell ?pushedCell - cell ?rock - rock)
    :precondition (and
      (at ?p ?playerCell)
      (rock-at ?rock ?rockCell)
      (empty ?pushedCell)
      (faced-left ?p)
      (adjacent-left ?playerCell ?rockCell)
      (adjacent-left ?rockCell ?pushedCell)
    )
    :effect (and
      (not (at ?p ?playerCell))
      (at ?p ?rockCell)
      (not (rock-at ?rock ?rockCell))
      (rock-at ?rock ?pushedCell)
      (not (empty ?pushedCell))
      (empty ?playerCell)
    )
  )
)
