# (C) 2001-2021 Intel Corporation. All rights reserved.
# Your use of Intel Corporation's design tools, logic functions and other 
# software and tools, and its AMPP partner logic functions, and any output 
# files from any of the foregoing (including device programming or simulation 
# files), and any associated documentation or information are expressly subject 
# to the terms and conditions of the Intel Program License Subscription 
# Agreement, Intel FPGA IP License Agreement, or other applicable 
# license agreement, including, without limitation, that your use is for the 
# sole purpose of programming logic devices manufactured by Intel and sold by 
# Intel or its authorized distributors.  Please refer to the applicable 
# agreement for further details.


#####################################################################
#
# THIS IS AN AUTO-GENERATED FILE!
# -------------------------------
# If you modify this files, all your changes will be lost if you
# regenerate the core!
#
# FILE DESCRIPTION
# ----------------
# This file contains the timing constraints for Intel Native AI Optimized DSP IP.

set script_dir [file dirname [info script]]
source "$script_dir/dsp_prime_parameters.tcl"

set ::GLOBAL_corename_debug 0
set ::GLOBAL_corename $var(output_name)

# ----------------------------------------------------------------
#
# Load required package
#
catch {
    load_package design
} err_loading_packages

# ----------------------------------------------------------------
#
proc ai_post_message {msg_type msg {msg_context sta_only}} {
#
# Description: Posts a message to Quartus, depending on 
# msg_context (sta_only, all)
#                           
# ----------------------------------------------------------------

    if {$msg_type == "debug"} {
        if {$::GLOBAL_corename_debug} {
            puts $msg
        }
    } else {
        if {$msg_context == "all"} {
            post_message -type $msg_type $msg
        } elseif {$msg_context == "sta_only"} {
            if {$::TimeQuestInfo(nameofexecutable) == "quartus_sta"} {
                post_message -type $msg_type $msg
            }
        }
    }
}


# ----------------------------------------------------------------
#
proc ai_get_core_full_instance_list {corename} {
#
# Description: Finds the instances of the particular IP by searching through the cells
#
# ----------------------------------------------------------------
    set instance_list [design::get_instances -entity $corename]
    return $instance_list
}



# ----------------------------------------------------------------
#
proc ai_get_core_instance_list {corename} {
#
# Description: Converts node names from one style to another style
#
# ----------------------------------------------------------------
    set full_instance_list [ai_get_core_full_instance_list $corename]
    set instance_list [list]

    foreach inst $full_instance_list {
        if {[lsearch $instance_list [escape_brackets $inst]] == -1} {
            ai_post_message debug "Found instance:  $inst"
            lappend instance_list $inst
        }
    }
    return $instance_list
}


# ----------------------------------------------------------------
#
proc ai_initialize_dsp_prime_db { dsp_prime_db_par var_array_name } {
#
# Description: Gets the instances of this particular DSP Prime IP and creates the pin
#              cache
#
# ----------------------------------------------------------------
    upvar $dsp_prime_db_par local_dsp_prime_db
    upvar 1 $var_array_name var

    global ::GLOBAL_corename

    ai_post_message info "Initializing DSP PRIME database for CORE $::GLOBAL_corename"
    set instance_list [ai_get_core_instance_list $::GLOBAL_corename]

    foreach instname $instance_list {
        ai_post_message info "Finding port-to-pin mapping for CORE: $::GLOBAL_corename INSTANCE: $instname"
        ai_get_dsp_prime_pins $instname allpins var
        set local_dsp_prime_db($instname) [ array get allpins ]
    }
}

# ----------------------------------------------------------------
#
proc ai_get_dsp_prime_pins { instname allpins var_array_name } {
#
# Description: Gets the pins of interest for this instance
#
# ----------------------------------------------------------------
    upvar allpins pins
    upvar 1 $var_array_name var
	
    set pins(src) "${instname}|fourteennm_dsp_prime_component|feed_sel[*]"
 
}

# Find required information including pin names
ai_initialize_dsp_prime_db dsp_prime_db_${::GLOBAL_corename} var
upvar 0 dsp_prime_db_${::GLOBAL_corename} local_db

set instances [ array names local_db ]
foreach { inst } $instances {
    if { [ info exists pins ] } {
        unset pins
    }
    array set pins $local_db($inst)
   
    # multicycle constraints
	set_multicycle_path -to [get_pins $pins(src)] -setup -end 2
	set_multicycle_path -to [get_pins $pins(src)] -hold  -end 1 
   
}