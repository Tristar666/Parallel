#!/bin/bash
for i in {800000..65000001..6320000}
   do
		        eval ./lab7 $i >> prof/test$i   
			echo "DONE $i"
   done
