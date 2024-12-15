import os
import pandas as pd
import numpy as np



class KeyResultCurveDataReader:

    def __init__( self, replications_directory_paths ):

        self.time_vals = np.array([])
        self.repl_dir_paths = replications_directory_paths
        self.replications_results = dict()
        self.channels_in_data = set()
        self.replications_in_data = set()

    def __parse_dat_file( self, channel_data_file, win_len ):

        try:
            df = pd.read_csv( channel_data_file, header=None, skiprows=8, delimiter=',', usecols=[1] )
            filtered_df = df.rolling( win_len, min_periods=1).mean()

            if self.time_vals.size == 0:
                df = pd.read_csv( channel_data_file, header=None, skiprows=8, delimiter=',', usecols=[0] )
                # print( df )
                self.time_vals = df[0].to_numpy()

            return filtered_df[1].to_numpy()
        except pd.errors.EmptyDataError:
            return np.array( [] )
        
    def __load_replicatios_results_from_single_directory( self, replications_directory_path, win_len=1 ):

        repl_names = os.listdir( replications_directory_path )
        replications_directory_name = os.path.basename( os.path.normpath( replications_directory_path ) )

        repl_results = dict()

        for repl_name in repl_names:

            repl_channels_dir = replications_directory_path + '/' + repl_name + '/' + 'PPO' + '/' + 'KeyResultCurve'

            channel_names = os.listdir( repl_channels_dir )

            for channel_name in channel_names:

                dat_file_path = repl_channels_dir + '/' + channel_name + '/' + channel_name + '.dat'

                if os.path.isfile( dat_file_path ):

                    time_series_data = self.__parse_dat_file( dat_file_path, win_len )
                    repl_results[ ( replications_directory_name, repl_name, channel_name ) ] = ( time_series_data.size, time_series_data )
                else:
                    repl_results[ ( replications_directory_name, repl_name, channel_name ) ] = ( 0, None )

                self.channels_in_data.add( channel_name )
                self.replications_in_data.add( ( replications_directory_name, repl_name ) )
        
        return repl_results
    
    def __replication_id( self, replication_kw ):

        return ( int( replication_kw[1][-3:] ), replication_kw[0] )
            
    def load_replications_results( self, moving_average_win_len=1 ):  # filter_function=None, filter_function_args=None

        for replications_directory_path in self.repl_dir_paths:

            single_rep_res = self.__load_replicatios_results_from_single_directory( replications_directory_path, moving_average_win_len )

            self.replications_results.update( single_rep_res )
    
    def get_time_values( self ):

        return self.time_vals
    
    def get_available_replications( self ):

        #return self.replications_in_data
        return sorted( self.replications_in_data, key=self.__replication_id )
    
    def get_available_channels( self ):

        return self.channels_in_data

    def get_all_replications_results( self ):

        results_without_file_size = dict()

        for key, item in self.replications_results.items():

            results_without_file_size[key] = item[1]
        
        return results_without_file_size
    
    def get_single_replication_results( self, repl_id ):

        single_replication_result = dict()

        for key in self.replications_results:

            if key[0:2] == repl_id:
                single_replication_result[key[2]] = self.replications_results[key][1]
        
        return single_replication_result

    def get_channels_with_missing_data( self ):

        channels_with_empty_data_file = list()

        for key, channel_data in self.replications_results.items():

            if channel_data[0] == 0:

                channels_with_empty_data_file.append( key )

        return channels_with_empty_data_file

    def remove_channels_from_all_replications( self, channel_names ):

        for channel_name in channel_names:
            if channel_name in self.channels_in_data:

                for key in list( self.replications_results ):

                    if key[2] == channel_name:

                        del self.replications_results[key]

                self.channels_in_data.remove( channel_name )
            else:
                print( f"Channel {channel_name} is not present in the data, therefore it can't be removed." )
    
    def remove_replications( self, replications ):

        '''
        replications: list of tuples, each tuple is 2 long and contains the main folder name and the simulation folder name, e.g., ( US, SO001 )
        '''

        for replication in replications:

            if replication in self.replications_in_data:

                for key in list( self.replications_results ):

                    if key[0:2] == replication:

                        del self.replications_results[key]

                self.replications_in_data.remove( replication )            
            else:
                print( f"Replication {replication} is not present in the data, therefore it can't be removed." )

    def __generate_2dnumpy_array_from_replication( self, replication_with_main_folder, channels_to_include ):

        repl_data = dict() # Will contain data for the specific replication with channel names as keys
        repl_np_array = list()

        for key in self.replications_results:

            if key[0:2] == replication_with_main_folder:

                repl_data[ key[2] ] = self.replications_results[ key ]

        smallest_array_length_in_replication = min( [ self.replications_results[key][0] for key in self.replications_results.keys() ] )

        for channel_name in channels_to_include:
            
            channel_array = repl_data[ channel_name ][1]

            if channel_array.size > smallest_array_length_in_replication:
                channel_array = channel_array[0:smallest_array_length_in_replication]
            
            repl_np_array.append( channel_array )
        
        repl_np_array = np.stack( repl_np_array, axis=0 )

        return repl_np_array


    
    ## Removing replications with not exactly 1400 sample points per channel. Append scalars to curves to be scaled together. ??
    def replications_with_not_1400_elements_removed(self):
        ## Removing replications with not exactly 1400 sample points per channel. Append scalars to curves to be scaled together.
        reps_removed = list()

        for rep_res_key in self.get_available_replications():

            rep_res = self.get_single_replication_results( rep_res_key ) # Reading the results (all channels) of a single simulation.

            lengths_not_ok = [ len(val) != 1400 for val in rep_res.values() ] # Checks the lengths of all columns (channels) of the simulation output channels.

            if any( lengths_not_ok ): # If any of the channels in the simulation result channels wasn't 1400 samples long, the simulation result gets removed from the database.
                self.remove_replications( [ rep_res_key ] )
                reps_removed.append( rep_res_key )
        

