(define (problem boulder_dash_problem)
  (:domain boulder_dash)

  ;; Objects
  (:objects
    p1 - player
    d1 d2 d3 d4 d5 d6 - diamond
    r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19 r20 r21 r22 r23 r24 r25 r26 r27 r28 r29 r30 r31 r32 r33 r34 r35 r36 r37 r38 r39 r40 r41 r42 r43 r44 r45 r46 r47 r48 r49 r50 r51 r52 r53 r54 r55 r56 r57 r58 r59 r60 r61 r62 r63 r64 r65 r66 r67 r68 r69 r70 r71 r72 r73 r74 r75 r76 r77 r78 r79 r80 r81 r82 r83 r84 r85 r86 r87 r88 r89 r90 r91 r92 r93 - rock
    c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c19 c20 c21 c22 c23 c24 c25 c26 c27 c28 c29 c30 c31 c32 c33 c34 c35 c36 c37 c38 c39 c40 c41 c42 c43 c44 c45 c46 c47 c48 c49 c50 c51 c52 c53 c54 c55 c56 c57 c58 c59 c60 c61 c62 c63 c64 c65 c66 c67 c68 c69 c70 c71 c72 c73 c74 c75 c76 c77 c78 c79 c80 c81 c82 c83 c84 c85 c86 c87 c88 c89 c90 c91 c92 c93 c94 c95 c96 c97 c98 c99 c100 c101 c102 c103 c104 c105 c106 c107 c108 c109 c110 c111 c112 c113 c114 c115 c116 c117 c118 c119 c120 c121 c122 c123 c124 c125 c126 c127 c128 c129 c130 c131 c132 c133 c134 c135 c136 c137 c138 c139 c140 c141 c142 c143 c144 c145 c146 c147 c148 c149 c150 c151 c152 c153 c154 c155 c156 c157 c158 c159 c160 c161 c162 c163 c164 c165 c166 c167 c168 c169 c170 c171 c172 c173 c174 c175 c176 c177 c178 c179 c180 c181 c182 c183 c184 c185 c186 c187 c188 c189 c190 c191 c192 c193 c194 c195 c196 c197 c198 c199 c200 c201 c202 c203 c204 c205 c206 c207 c208 c209 c210 c211 c212 c213 c214 c215 c216 c217 c218 c219 c220 c221 c222 c223 c224 c225 c226 c227 c228 c229 c230 c231 c232 c233 c234 c235 c236 c237 c238 c239 c240 c241 c242 c243 c244 c245 c246 c247 c248 c249 c250 c251 c252 c253 c254 c255 c256 c257 c258 c259 c260 c261 c262 c263 c264 c265 c266 c267 c268 c269 c270 c271 c272 c273 c274 c275 c276 c277 c278 c279 c280 c281 c282 c283 c284 c285 c286 c287 c288 c289 c290 c291 c292 c293 c294 c295 c296 c297 c298 c299 c300 c301 c302 c303 c304 c305 c306 c307 c308 c309 c310 c311 c312 c313 c314 c315 c316 c317 c318 c319 c320 c321 c322 c323 c324 c325 c326 c327 c328 c329 c330 c331 c332 c333 c334 c335 c336 c337 c338 c339 c340 c341 c342 c343 c344 c345 c346 c347 c348 c349 c350 c351 c352 c353 c354 c355 c356 c357 c358 c359 c360 c361 c362 c363 c364 c365 c366 c367 c368 c369 c370 c371 c372 c373 c374 c375 c376 c377 c378 c379 c380 c381 c382 c383 c384 c385 c386 c387 c388 c389 c390 c391 c392 c393 c394 c395 c396 c397 c398 c399 c400 c401 c402 c403 c404 c405 c406 c407 c408 c409 c410 c411 c412 c413 c414 c415 c416 c417 c418 c419 c420 c421 c422 c423 c424 c425 c426 c427 c428 c429 c430 c431 c432 c433 c434 c435 c436 c437 c438 c439 c440 c441 c442 c443 c444 c445 c446 c447 c448 c449 c450 c451 c452 c453 c454 c455 c456 c457 c458 c459 c460 c461 c462 c463 c464 c465 c466 c467 c468 c469 c470 c471 c472 c473 c474 c475 c476 c477 c478 c479 c480 c481 c482 c483 c484 c485 c486 c487 c488 c489 c490 c491 c492 c493 c494 c495 c496 c497 c498 c499 c500 c501 c502 c503 c504 c505 c506 c507 c508 c509 c510 c511 c512 c513 c514 c515 c516 c517 c518 c519 c520 c521 c522 c523 c524 c525 c526 c527 c528 c529 c530 c531 c532 c533 c534 c535 c536 c537 c538 c539 c540 c541 c542 c543 c544 c545 c546 c547 c548 c549 c550 c551 c552 c553 c554 c555 c556 c557 c558 c559 c560 c561 c562 c563 c564 c565 c566 c567 c568 c569 c570 c571 c572 c573 c574 c575 c576 c577 c578 c579 c580 c581 c582 c583 c584 c585 c586 c587 c588 c589 c590 c591 c592 c593 c594 c595 c596 c597 c598 c599 c600 c601 c602 c603 c604 c605 c606 c607 c608 c609 c610 c611 c612 c613 c614 c615 c616 c617 c618 c619 c620 c621 c622 c623 c624 c625 c626 c627 c628 c629 c630 c631 c632 c633 c634 c635 c636 c637 c638 c639 c640 c641 c642 c643 c644 c645 c646 c647 c648 c649 c650 c651 c652 c653 c654 c655 c656 c657 c658 c659 c660 c661 c662 c663 c664 c665 c666 c667 c668 c669 c670 c671 c672 c673 c674 c675 c676 c677 c678 c679 c680 c681 c682 c683 c684 c685 c686 c687 c688 c689 c690 c691 c692 c693 c694 c695 c696 c697 c698 c699 c700 c701 c702 c703 c704 c705 c706 c707 c708 c709 c710 c711 c712 c713 c714 c715 c716 c717 c718 c719 c720 c721 c722 c723 c724 c725 c726 c727 c728 c729 c730 c731 c732 c733 c734 c735 c736 c737 c738 c739 c740 c741 c742 c743 c744 c745 c746 c747 c748 c749 c750 c751 c752 c753 c754 c755 c756 c757 c758 c759 c760 c761 c762 c763 c764 c765 c766 c767 c768 c769 c770 c771 c772 c773 c774 c775 c776 c777 c778 c779 c780 c781 c782 c783 c784 c785 c786 c787 c788 c789 c790 c791 c792 c793 c794 c795 c796 c797 c798 c799 c800 c801 c802 c803 c804 c805 c806 c807 c808 c809 c810 c811 c812 c813 c814 c815 c816 c817 c818 c819 c820 c821 c822 c823 c824 c825 c826 c827 c828 c829 c830 c831 c832 c833 c834 c835 c836 c837 c838 c839 c840 c841 c842 c843 c844 c845 c846 c847 c848 c849 c850 c851 c852 c853 c854 c855 c856 c857 c858 c859 c860 c861 c862 c863 c864 c865 c866 c867 c868 c869 c870 c871 c872 c873 c874 c875 c876 c877 c878 c879 c880 c881 c882 c883 c884 c885 c886 c887 c888 c889 c890 c891 c892 c893 c894 c895 c896 c897 c898 c899 c900 - cell
  )

  ;; Initial State
  (:init
    (at gem1 c_5_3)
    (connected−up c_5_3 c_5_2)
    (connected−down c_5_3 c_5_4)
    (connected−right c_5_3 c_6_3)
    (connected−left c_5_3 c_4_3)
  )

  (:goal
    ( got gem_1_3)
  )
)