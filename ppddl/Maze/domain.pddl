(define (domain Maze)

(:requirements :typing)

(:types vehicle square)

(:predicates
  (at ?x - vehicle ?y - square) 
  
)

	
(:action North
:parameters (?x - rover ?y - waypoint ?z - waypoint) 
:precondition (and (can_traverse ?x ?y ?z) (available ?x) (at ?x ?y) 
                (visible ?y ?z)
	    )
:effect (and (not (at ?x ?y)) (at ?x ?z)
		)
)


)
